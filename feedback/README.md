# feedback/

This directory is intentionally empty at first. Once you log an event
outcome in the app's "Feedback Loop" tab, `outcomes_log.csv` is created
here automatically - that's the post-event learning system described in
the brief ("no post-event learning system" -> this closes that gap).

Nothing to set up manually; `src/feedback_store.py` creates the file on
first use.
