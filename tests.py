"""
tests.py — Unit tests for the Email Triage environment.
Run with: python tests.py
"""

import sys

from environment import (
    EmailTriageEnv,
    Action,
    grade_task1,
    grade_task2,
    InboxState,
    Email,
    TASK1_GROUND_TRUTH,
    TASK1_EMAILS
)

def run_test(name: str, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except AssertionError as e:
        print(f"  ❌ {name}: {e}")
        return False
    except Exception as e:
        print(f"  💥 {name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------

def test_task1_reset():
    env = EmailTriageEnv(task=1)
    obs = env.reset()
    assert obs.status == "ok"
    assert obs.data["inbox_size"] == 5

def test_task1_list():
    env = EmailTriageEnv(task=1)
    env.reset()
    result = env.step(Action(action="list_inbox"))
    assert result.observation.status == "ok"
    assert len(result.observation.data["emails"]) == 5

def test_task1_read():
    env = EmailTriageEnv(task=1)
    env.reset()
    result = env.step(Action(action="read", email_id="t1_001"))
    assert result.observation.status == "ok"
    assert len(result.observation.data["subject"]) > 0

def test_task1_label_correct():
    env = EmailTriageEnv(task=1)
    env.reset()
    gt = TASK1_GROUND_TRUTH["t1_001"]
    result = env.step(Action(action="label", email_id="t1_001", priority=gt))
    assert result.reward == 0.2, f"Expected 0.2, got {result.reward}"

def test_task1_label_wrong():
    env = EmailTriageEnv(task=1)
    env.reset()
    gt = TASK1_GROUND_TRUTH["t1_001"]
    wrong = "low" if gt in ("urgent", "normal") else "urgent"
    result = env.step(Action(action="label", email_id="t1_001", priority=wrong))
    assert result.reward == 0.0

def test_task1_full_score():
    env = EmailTriageEnv(task=1)
    env.reset()
    for eid, priority in TASK1_GROUND_TRUTH.items():
        env.step(Action(action="label", email_id=eid, priority=priority))
    assert env.score() == 1.0, f"Expected 1.0, got {env.score()}"

def test_task1_partial_score():
    env = EmailTriageEnv(task=1)
    env.reset()
    eids = list(TASK1_GROUND_TRUTH.keys())
    env.step(Action(action="label", email_id=eids[0], priority=TASK1_GROUND_TRUTH[eids[0]]))
    env.step(Action(action="label", email_id=eids[1], priority=TASK1_GROUND_TRUTH[eids[1]]))
    score = env.score()
    assert score == 0.4, f"Expected 0.4, got {score}"


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------

def test_task2_reset():
    env = EmailTriageEnv(task=2)
    obs = env.reset()
    assert obs.data["inbox_size"] == 1

def test_task2_no_reply_zero():
    env = EmailTriageEnv(task=2)
    env.reset()
    assert env.score() == 0.0

def test_task2_good_reply():
    env = EmailTriageEnv(task=2)
    env.reset()
    env.step(Action(
        action="draft_reply",
        email_id="t2_001",
        body=(
            "Dear Jamie,\n\nThank you for reaching out. We sincerely apologize for the "
            "experience you have had with order #48291. We understand how frustrating "
            "this must be.\n\nWe are urgently investigating the status of your delivery "
            "and will provide an update within 2 hours. If we cannot confirm delivery "
            "within 48 hours we will process a full refund immediately. We will also "
            "review the service failures you experienced and follow up regarding "
            "compensation.\n\nWe truly value your business and are committed to "
            "making this right.\n\nSincerely,\nCustomer Support Team"
        ),
    ))
    score = env.score()
    assert score > 0.5, f"Expected score > 0.5, got {score}"

def test_task2_short_reply_penalised():
    env = EmailTriageEnv(task=2)
    env.reset()
    result = env.step(Action(action="draft_reply", email_id="t2_001", body="ok"))
    assert result.observation.status == "error"


# ---------------------------------------------------------------------------
# Task 3 tests
# ---------------------------------------------------------------------------

def test_task3_reset():
    env = EmailTriageEnv(task=3)
    obs = env.reset()
    assert obs.data["inbox_size"] == 10

def test_task3_archive_spam_no_penalty():
    env = EmailTriageEnv(task=3)
    env.reset()
    # Label spam as low first (so archiving doesn't trigger urgent penalty)
    env.step(Action(action="label", email_id="t3_002", priority="low"))
    result = env.step(Action(action="archive", email_id="t3_002"))
    assert result.observation.status == "ok"

def test_task3_archive_urgent_penalty():
    env = EmailTriageEnv(task=3)
    env.reset()
    env.step(Action(action="label", email_id="t3_001", priority="urgent"))
    result = env.step(Action(action="archive", email_id="t3_001"))
    assert result.reward == -0.1
    assert result.observation.status == "warning"

def test_task3_flag():
    env = EmailTriageEnv(task=3)
    env.reset()
    result = env.step(Action(action="flag", email_id="t3_009", reason="Missing context — need sender identity"))
    assert result.observation.status == "ok"

def test_task3_loop_detection():
    env = EmailTriageEnv(task=3)
    env.reset()
    for _ in range(3):
        env.step(Action(action="label", email_id="t3_006", priority="normal"))
    assert env._penalties["loop_actions"] >= 1

def test_task3_not_found():
    env = EmailTriageEnv(task=3)
    env.reset()
    result = env.step(Action(action="read", email_id="nonexistent"))
    assert result.observation.status == "error"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Task 1
        ("Task1 reset", test_task1_reset),
        ("Task1 list inbox", test_task1_list),
        ("Task1 read email", test_task1_read),
        ("Task1 correct label reward", test_task1_label_correct),
        ("Task1 wrong label no reward", test_task1_label_wrong),
        ("Task1 full score 1.0", test_task1_full_score),
        ("Task1 partial score 0.4", test_task1_partial_score),
        # Task 2
        ("Task2 reset", test_task2_reset),
        ("Task2 no reply = 0.0", test_task2_no_reply_zero),
        ("Task2 good reply > 0.5", test_task2_good_reply),
        ("Task2 short reply error", test_task2_short_reply_penalised),
        # Task 3
        ("Task3 reset", test_task3_reset),
        ("Task3 archive spam no penalty", test_task3_archive_spam_no_penalty),
        ("Task3 archive urgent = penalty", test_task3_archive_urgent_penalty),
        ("Task3 flag ambiguous", test_task3_flag),
        ("Task3 loop detection", test_task3_loop_detection),
        ("Task3 not found error", test_task3_not_found),
    ]

    print("\nRunning Email Triage Environment Tests")
    print("=" * 45)
    passed = sum(run_test(name, fn) for name, fn in tests)
    total = len(tests)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
