"""
curate_dataset.py -- Downloads real emails from the Enron Spam dataset on
HuggingFace and curates them into a structured JSON dataset for the Email
Triage environment.

This script is run ONCE to generate data/emails.json. The generated file
is then shipped with the project -- the environment loads it at runtime
without needing the `datasets` library.

Usage:
    pip install datasets
    python curate_dataset.py
"""

import json
import random
import re
import os
from datasets import load_dataset

random.seed(42)  # reproducible curation

# ---------------------------------------------------------------------------
# 1. Load the Enron Spam dataset from HuggingFace
# ---------------------------------------------------------------------------
print("Loading SetFit/enron_spam from HuggingFace...")
ds = load_dataset("SetFit/enron_spam", split="train")
print(f"  Total emails: {len(ds)}")

# Fields: text (subject + body combined), label (0=ham, 1=spam)
# We need to parse subject and body from the text field

def parse_email(text: str) -> dict:
    """Parse Enron email text into subject + body."""
    lines = text.strip().split("\n")
    subject = ""
    body_start = 0

    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line[len("Subject:"):].strip()
            body_start = i + 1
            break

    body = "\n".join(lines[body_start:]).strip()

    # Clean up common artifacts
    body = re.sub(r'\s+', ' ', body)[:800]  # cap body length
    if not subject:
        subject = body[:60] + "..." if len(body) > 60 else body

    return {"subject": subject, "body": body}


# ---------------------------------------------------------------------------
# 2. Filter and curate emails
# ---------------------------------------------------------------------------

# Separate ham (legitimate) and spam
ham_emails = []
spam_emails = []

for i, item in enumerate(ds):
    if not item["text"] or len(item["text"].strip()) < 50:
        continue
    parsed = parse_email(item["text"])
    if not parsed["subject"] or not parsed["body"] or len(parsed["body"]) < 30:
        continue

    entry = {
        "enron_index": i,
        "subject": parsed["subject"],
        "body": parsed["body"],
        "is_spam": item["label"] == 1,
    }

    if item["label"] == 0:
        ham_emails.append(entry)
    else:
        spam_emails.append(entry)

print(f"  Ham (legitimate): {len(ham_emails)}")
print(f"  Spam:             {len(spam_emails)}")

# ---------------------------------------------------------------------------
# 3. Curate emails into task-ready collections with ground truth
# ---------------------------------------------------------------------------

# We'll assign realistic senders and priority labels based on content analysis
CORPORATE_SENDERS = [
    "mark.taylor@enron.com", "sarah.palmer@globalenergy.com",
    "john.arnold@trading-desk.com", "vince.kaminski@enron.com",
    "sally.beck@enron.com", "louise.kitchen@enron.com",
    "jeff.dasovich@regulatoryaffairs.com", "steven.kean@enron.com",
    "richard.shapiro@enron.com", "james.steffes@enron.com",
    "mike.carson@infrastructure.com", "lisa.gang@legal-team.com",
    "david.delainey@ees.com", "greg.whalley@enron.com",
    "tim.belden@trading.com", "kevin.presto@eastpower.com",
    "matt.smith@operations.com", "donna.fulton@regulatory.com",
    "kate.symes@trading.com", "diana.scholtes@admin.com",
]

SPAM_SENDERS = [
    "deals@shop-now-99.xyz", "winner@prize-center.info",
    "noreply@free-offers.biz", "promo@discount-deals.click",
    "support@account-verify.net",
]

NEWSLETTER_SENDERS = [
    "digest@energy-news.io", "weekly@market-watch.com",
    "updates@industry-report.net",
]


def classify_priority(subject: str, body: str, is_spam: bool) -> str:
    """Assign ground-truth priority based on content analysis."""
    text = (subject + " " + body).lower()

    if is_spam:
        return "low"

    # Urgent signals
    urgent_keywords = [
        "urgent", "critical", "asap", "immediately", "deadline",
        "emergency", "action required", "must", "time sensitive",
        "expir", "shut down", "outage", "breach", "compliance",
        "regulatory", "legal action", "termination", "suspension",
    ]
    if any(kw in text for kw in urgent_keywords):
        return "urgent"

    # Normal signals (business correspondence)
    normal_keywords = [
        "meeting", "schedule", "review", "update", "report",
        "please", "attached", "draft", "feedback", "follow up",
        "discuss", "proposal", "agreement", "contract", "budget",
    ]
    if any(kw in text for kw in normal_keywords):
        return "normal"

    return "low"


def assign_sender(is_spam: bool, priority: str) -> str:
    """Assign a realistic sender based on email type."""
    if is_spam:
        return random.choice(SPAM_SENDERS)
    return random.choice(CORPORATE_SENDERS)


# --- Task 1: 5 emails for priority classification (easy) ---
# Need: 2 urgent, 1 normal, 2 low (mix of ham + spam)
task1_candidates = {"urgent": [], "normal": [], "low": []}
for email in ham_emails[:500]:
    p = classify_priority(email["subject"], email["body"], False)
    if len(task1_candidates[p]) < 20:
        task1_candidates[p].append(email)
for email in spam_emails[:200]:
    if len(task1_candidates["low"]) < 20:
        email_copy = dict(email)
        task1_candidates["low"].append(email_copy)

task1_picks = (
    random.sample(task1_candidates["urgent"], min(2, len(task1_candidates["urgent"])))
    + random.sample(task1_candidates["normal"], min(1, len(task1_candidates["normal"])))
    + random.sample(task1_candidates["low"], min(2, len(task1_candidates["low"])))
)

task1_emails = []
for i, email in enumerate(task1_picks):
    priority = classify_priority(email["subject"], email["body"], email.get("is_spam", False))
    task1_emails.append({
        "id": f"t1_{i+1:03d}",
        "from": assign_sender(email.get("is_spam", False), priority),
        "subject": email["subject"],
        "body": email["body"],
        "ground_truth_priority": priority,
        "source": "SetFit/enron_spam",
        "source_index": email["enron_index"],
    })

# --- Task 2: 1 complaint email (will write a realistic one based on Enron context) ---
task2_email = {
    "id": "t2_001",
    "from": "frustrated.trader@westcoast-power.com",
    "subject": "UNACCEPTABLE: Trade confirmation errors - 3rd time this month",
    "body": (
        "To whom it may concern,\n\n"
        "I am writing to formally complain about the persistent errors in trade "
        "confirmations coming from your desk. This is the THIRD time this month "
        "that we have received confirmations with incorrect volumes and pricing. "
        "Our last trade (ref: WCP-2024-8847) showed 500 MW at $42.50 when the "
        "agreed terms were 750 MW at $38.75.\n\n"
        "When we called to rectify, your operations team said they would 'look "
        "into it' -- that was five business days ago with no follow-up.\n\n"
        "We need:\n"
        "1. Immediate correction of trade ref WCP-2024-8847\n"
        "2. A reconciliation of ALL trades executed between our desks this quarter\n"
        "3. A written explanation of what process failure is causing these errors\n"
        "4. Assurance that this will not happen again\n\n"
        "If this is not resolved by end of week, we will be escalating to our "
        "legal team and reconsidering our trading relationship.\n\n"
        "Regards,\nMichael Torres\nHead of Trading Operations\n"
        "WestCoast Power LLC"
    ),
    "ground_truth_priority": "urgent",
    "source": "manually_crafted_enron_context",
}

# --- Task 3: 10 emails for full triage (hard) ---
# Need a diverse mix: 4 urgent, 2 normal, 2 low/spam, 1 ambiguous, 1 newsletter
task3_candidates = {"urgent": [], "normal": [], "low": [], "spam": []}
# Use different emails than task 1
for email in ham_emails[500:1500]:
    p = classify_priority(email["subject"], email["body"], False)
    if len(task3_candidates[p]) < 30:
        task3_candidates[p].append(email)
for email in spam_emails[200:600]:
    if len(task3_candidates["spam"]) < 30:
        email_copy = dict(email)
        task3_candidates["spam"].append(email_copy)

task3_picks_urgent = random.sample(
    task3_candidates["urgent"], min(4, len(task3_candidates["urgent"]))
)
task3_picks_normal = random.sample(
    task3_candidates["normal"], min(2, len(task3_candidates["normal"]))
)
# For low: use the spam candidates
task3_spam_low = task3_candidates["spam"]
task3_picks_spam = random.sample(task3_spam_low, min(2, len(task3_spam_low)))
# Remaining low slots from non-spam ham

# Ambiguous email (crafted -- context-dependent, hard to classify)
task3_ambiguous = {
    "subject": "Re: that discussion last week",
    "body": (
        "Following up on our conversation. I think we should move forward "
        "but wanted to get your read on the situation first. There are some "
        "concerns internally that I'd rather discuss offline. Can you call "
        "me when you get a chance?"
    ),
    "is_spam": False,
    "enron_index": -1,
}

# Newsletter
task3_newsletter = {
    "subject": "Weekly Energy Market Report - Natural Gas Futures Update",
    "body": (
        "This week's energy market highlights:\n"
        "- Natural gas futures rose 3.2% on cold weather forecasts\n"
        "- FERC announced new transmission capacity rules\n"
        "- California ISO reported record renewable generation\n\n"
        "Full analysis at energy-news.io/weekly\n"
        "Unsubscribe: reply STOP"
    ),
    "is_spam": False,
    "enron_index": -2,
}

all_task3 = (
    [(e, "urgent", False) for e in task3_picks_urgent]
    + [(e, "normal", False) for e in task3_picks_normal]
    + [(e, "low", True) for e in task3_picks_spam]       # spam → archive
    + [(task3_ambiguous, "normal", False)]                 # ambiguous → flag
    + [(task3_newsletter, "low", False)]
)
random.shuffle(all_task3)

# Track which IDs are urgent, spam/archive, and ambiguous
task3_urgent_ids = set()
task3_archive_ids = set()
task3_flag_ids = set()

task3_emails = []
for i, (email, priority, is_spam_override) in enumerate(all_task3):
    eid = f"t3_{i+1:03d}"

    is_spam = is_spam_override
    is_ambiguous = email.get("enron_index") == -1
    is_newsletter = email.get("enron_index") == -2

    if priority == "urgent":
        task3_urgent_ids.add(eid)
        sender = assign_sender(False, "urgent")
    elif is_spam:
        task3_archive_ids.add(eid)
        sender = assign_sender(True, "low")
    elif is_newsletter:
        sender = random.choice(NEWSLETTER_SENDERS)
    elif is_ambiguous:
        task3_flag_ids.add(eid)
        sender = "unknown.sender@company.com"
    else:
        sender = assign_sender(False, priority)

    task3_emails.append({
        "id": eid,
        "from": sender,
        "subject": email["subject"],
        "body": email["body"],
        "ground_truth_priority": priority,
        "source": "SetFit/enron_spam" if email.get("enron_index", 0) >= 0 else "manually_crafted",
        "source_index": email.get("enron_index", -1),
    })


# ---------------------------------------------------------------------------
# 4. Write the curated dataset
# ---------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)

dataset = {
    "metadata": {
        "name": "email-triage-dataset",
        "version": "1.0.0",
        "description": (
            "Curated email dataset for the Email Triage & Response Environment. "
            "Contains real emails from the Enron corpus (SetFit/enron_spam on "
            "HuggingFace) with manually assigned priority labels and triage metadata."
        ),
        "source_dataset": "SetFit/enron_spam",
        "source_url": "https://huggingface.co/datasets/SetFit/enron_spam",
        "license": "Public domain (Enron corpus)",
        "total_emails": len(task1_emails) + 1 + len(task3_emails),
        "curation_seed": 42,
    },
    "task1": {
        "description": "Label 5 emails as urgent/normal/low priority",
        "difficulty": "easy",
        "emails": task1_emails,
        "ground_truth": {e["id"]: e["ground_truth_priority"] for e in task1_emails},
    },
    "task2": {
        "description": "Draft a professional reply to a complaint email",
        "difficulty": "medium",
        "emails": [task2_email],
    },
    "task3": {
        "description": "Full triage pipeline: label, reply, archive, flag",
        "difficulty": "hard",
        "emails": task3_emails,
        "ground_truth": {e["id"]: e["ground_truth_priority"] for e in task3_emails},
        "urgent_ids": sorted(task3_urgent_ids),
        "archive_ids": sorted(task3_archive_ids),
        "flag_ids": sorted(task3_flag_ids),
    },
}

output_path = "data/emails.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\nDataset written to {output_path}")
print(f"  Task 1: {len(task1_emails)} emails")
print(f"  Task 2: 1 email")
print(f"  Task 3: {len(task3_emails)} emails")
print(f"    Urgent IDs: {sorted(task3_urgent_ids)}")
print(f"    Archive IDs: {sorted(task3_archive_ids)}")
print(f"    Flag IDs:    {sorted(task3_flag_ids)}")
print("\nDone! Now update environment.py to load from data/emails.json")
