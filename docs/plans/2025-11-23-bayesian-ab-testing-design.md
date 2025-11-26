# Bayesian A/B Testing Design

**Date:** 2025-11-23
**Framework:** Facebook Ax (Adaptive Experimentation)
**Target:** Curriculum variation optimization for IGA language learning app
**Status:** Design approved, ready for implementation

---

## Executive Summary

This design integrates Facebook's Ax framework for Bayesian A/B testing into the IGA Gradio app. The system will test 4 curriculum dimensions (phrase ordering, category focus, repetition strategy, difficulty progression) across 36 possible variants, optimizing for 3 metrics (engagement, retention, user satisfaction) using dynamic arm selection and real-time posterior updates.

**Key benefit:** Transparent, adaptive curriculum optimization that improves learning outcomes without disrupting user experience.

---

## Requirements

### Optimization Targets
- **Curriculum dimensions to test:**
  - Phrase ordering: random vs. difficulty-ascending vs. difficulty-descending
  - Category focus: mixed categories vs. specialized category focus
  - Repetition strategy: immediate vs. spaced-1h vs. spaced-24h
  - Difficulty progression: easy-first vs. varied mix
- **Total variants:** 3 × 2 × 3 × 2 = 36 possible arm combinations

### Metrics to Optimize
1. **Engagement Score** (0-1): Session duration + phrases attempted (normalized)
2. **Retention Score** (0-1): XP earned + session consistency (normalized)
3. **Satisfaction Score** (0-1): User rating (1-5 scale) or inferred from engagement

### Constraints
- Deployment: Hugging Face Spaces + Gradio
- Data storage: DuckDB (lightweight, SQL-based, no external dependencies)
- No external analytics services
- User privacy: Anonymized session IDs only, no PII collection
- Optimization runs incrementally every N phrases (~10)

---

## Architecture Overview

### High-Level Flow

```
User Session Starts
    ↓
Check/Assign Variant (DuckDB)
    ↓
For each phrase in session:
  - Query Ax for next arm (curriculum parameters)
  - Fetch phrase from corpus with selected parameters
  - User completes phrase
  - Log metrics to DuckDB
  - Every 10 phrases: Update Ax posterior
    ↓
Session ends → Log summary metrics → Optional: Collect user rating
```

### Core Principle: Integrated Adaptive Optimization

- **Dynamic arm selection:** Each phrase selection uses current posterior distribution
- **Real-time learning:** Metrics logged immediately, posterior updated incrementally
- **Epsilon-greedy exploration:** 80% exploit best variants, 20% explore for discovery
- **Multi-objective balancing:** Weighted sum scalarizes 3 metrics into single optimization objective

---

## Component Design

### 1. `bayesian_optimizer.py` (New)

**Purpose:** Wraps Ax framework and manages Bayesian optimization.

**Key Classes:**

```python
class AxOptimizer:
    def __init__(self, metrics_weights: Dict[str, float]):
        """
        Initialize Ax experiment with 4 parameters, 3 metrics.

        Args:
            metrics_weights: {engagement: 0.4, retention: 0.4, satisfaction: 0.2}
        """

    def get_next_arm(self) -> Dict[str, str]:
        """
        Select next variant based on current posterior.

        Returns: {
            phrase_ordering: 'random',
            category_focus: 'mixed',
            repetition_strategy: 'spaced-24h',
            difficulty_balance: 'varied-mix'
        }

        Strategy: 80% posterior mean, 20% random exploration
        """

    def update_posterior(self, metrics_data: List[Dict]) -> None:
        """
        Read DuckDB metrics, run Bayesian analysis, update posterior.

        Args:
            metrics_data: List of trial outcomes from DuckDB

        Side effect: Updates internal Ax experiment state
        """

    def get_arm_probabilities(self) -> Dict[str, float]:
        """Return probability of selecting each variant (for monitoring)."""

    def get_best_arm(self) -> Dict[str, str]:
        """Return highest posterior mean variant."""
```

**Behavior:**
- Lazy-loads Ax experiment from DuckDB (or creates new)
- Caches experiment object to avoid recomputation
- Uses Thompson sampling or posterior sampling for arm selection
- Handles sparse data gracefully (weak priors for early trials)

---

### 2. `ab_test_logging.py` (New)

**Purpose:** DuckDB-based metric collection and event logging.

**Key Classes:**

```python
class ABTestLogger:
    def __init__(self, db_path: str = "ab_test.db"):
        """Initialize DuckDB connection, create schema if missing."""

    def log_phrase_attempt(
        self,
        user_id: str,
        variant_id: str,
        phrase: str,
        duration_ms: int,
        success: bool
    ) -> None:
        """Log individual phrase attempt."""

    def log_session_end(
        self,
        user_id: str,
        variant_id: str,
        total_xp: int,
        session_duration_min: float
    ) -> None:
        """Log session summary (engagement, retention metrics)."""

    def log_user_feedback(
        self,
        user_id: str,
        variant_id: str,
        rating: int  # 1-5 scale
    ) -> None:
        """Log explicit user satisfaction rating."""

    def get_metrics_for_optimization(self) -> List[Dict]:
        """
        Return formatted metrics for Ax optimization.
        Aggregates per-user-variant metrics for Bayesian analysis.
        """

    def export_to_csv(self, path: str) -> None:
        """Export all metrics to CSV for external analysis."""
```

**DuckDB Schema:**

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    current_variant_id TEXT,
    created_at TIMESTAMP,
    last_active TIMESTAMP
);

CREATE TABLE phrase_attempts (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    variant_id TEXT,
    phrase TEXT,
    duration_ms INTEGER,
    success BOOLEAN,
    timestamp TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    variant_id TEXT,
    total_xp INTEGER,
    session_duration_min FLOAT,
    completed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    variant_id TEXT,
    rating INTEGER,  -- 1-5
    timestamp TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

---

### 3. `variant_manager.py` (New)

**Purpose:** Manage user-variant assignments and state persistence.

**Key Classes:**

```python
class VariantManager:
    def __init__(self, db_logger: ABTestLogger):
        """Initialize with DuckDB logger for persistence."""

    def get_or_assign_user_variant(
        self,
        user_id: str,
        optimizer: AxOptimizer
    ) -> Dict[str, str]:
        """
        Return variant for user.
        - If new user: Assign random initial variant
        - If returning user: Return current variant from DuckDB

        Returns variant dict: {phrase_ordering, category_focus, ...}
        """

    def update_user_variant(
        self,
        user_id: str,
        new_variant: Dict[str, str]
    ) -> None:
        """Switch user to new variant (adaptive mid-session)."""

    def adaptive_variant_switch(
        self,
        user_id: str,
        current_performance: float,
        median_performance: float,
        optimizer: AxOptimizer
    ) -> Optional[Dict[str, str]]:
        """
        If user performance << median, suggest switching to better-performing variant.
        Returns new variant if switch recommended, else None.
        User can accept/decline (respects user agency).
        """
```

---

### 4. Integration into `app.py`

**Startup (in `main()`):**

```python
# Initialize A/B testing components
logger = ABTestLogger(db_path="ab_test.db")
optimizer = AxOptimizer(metrics_weights={
    "engagement": 0.4,
    "retention": 0.4,
    "satisfaction": 0.2
})
variant_manager = VariantManager(logger)

# Load initial posterior from DuckDB (from previous runs)
initial_metrics = logger.get_metrics_for_optimization()
if initial_metrics:
    optimizer.update_posterior(initial_metrics)

# Start background task for periodic posterior updates
@spaces.run_fn  # or standard async task
def periodic_optimizer_update():
    while True:
        sleep(300)  # Every 5 minutes
        metrics = logger.get_metrics_for_optimization()
        optimizer.update_posterior(metrics)
```

**In learning mode functions** (e.g., `pronunciation_feedback()` or `translation_quiz()`):

```python
def pronunciation_feedback(audio_input, text_input):
    # 1. Get user ID (session-based anonymized ID)
    user_id = get_session_user_id()

    # 2. Get or assign variant
    variant = variant_manager.get_or_assign_user_variant(user_id, optimizer)

    # 3. For each phrase in session:
    session_xp = 0
    session_start = time.time()

    for i in range(num_phrases):
        # Get next arm for this phrase
        arm = optimizer.get_next_arm()

        # Fetch phrase with curriculum parameters
        phrase = corpus.get_phrase(
            category_focus=arm["category_focus"],
            ordering=arm["phrase_ordering"],
            difficulty=arm["difficulty_balance"]
        )

        # User completes phrase (existing logic)
        xp_earned, success = process_phrase(audio_input, phrase)
        session_xp += xp_earned

        # Log phrase-level metrics
        duration = time.time() - phrase_start
        logger.log_phrase_attempt(
            user_id=user_id,
            variant_id=str(variant),
            phrase=phrase,
            duration_ms=int(duration * 1000),
            success=success
        )

        # Every 10 phrases, update posterior
        if (i + 1) % 10 == 0:
            metrics = logger.get_metrics_for_optimization()
            optimizer.update_posterior(metrics)

            # Optional: Suggest variant switch if performance is poor
            current_perf = calculate_session_performance(...)
            median_perf = logger.get_median_performance()
            new_variant = variant_manager.adaptive_variant_switch(
                user_id, current_perf, median_perf, optimizer
            )
            if new_variant:
                # Show user suggestion UI (optional Gradio component)
                pass

    # 4. Log session summary
    session_duration = (time.time() - session_start) / 60
    logger.log_session_end(
        user_id=user_id,
        variant_id=str(variant),
        total_xp=session_xp,
        session_duration_min=session_duration
    )

    # 5. Optional: Request user rating
    return result, gr.update(visible=True)  # Show rating component
```

**Admin Panel (new Gradio tab):**

```python
with gr.Tab("A/B Test Results"):
    gr.Markdown("### Bayesian Optimization Status")

    with gr.Row():
        arm_prob_plot = gr.Plot(
            value=plot_arm_probabilities(optimizer.get_arm_probabilities())
        )
        best_arm_text = gr.Textbox(
            value=format_arm(optimizer.get_best_arm()),
            label="Best Variant (Highest Posterior Mean)",
            interactive=False
        )

    metrics_table = gr.Dataframe(
        value=logger.get_metrics_summary(),
        label="Metric Summary by Variant"
    )

    export_button = gr.Button("Export Metrics to CSV")
    export_button.click(
        fn=lambda: logger.export_to_csv("ab_test_export.csv"),
        outputs=gr.Textbox(value="Export complete")
    )
```

---

## Bayesian Optimization Details

### Experiment Definition

**Parameters (categorical):**
1. `phrase_ordering`: {random, difficulty-asc, difficulty-desc}
2. `category_focus`: {mixed, specialized}
3. `repetition_strategy`: {immediate, spaced-1h, spaced-24h}
4. `difficulty_balance`: {easy-first, varied-mix}

**Total arms:** 36 possible combinations

### Metrics and Objectives

Ax tracks 3 metrics, combined via weighted sum scalarization:

```python
objective = (
    0.4 * engagement_score +
    0.4 * retention_score +
    0.2 * satisfaction_score
)
```

**Metric definitions:**
- `engagement_score` = (session_duration_min / max_duration) * (phrases_attempted / max_phrases)
- `retention_score` = (xp_earned / expected_xp) * (1 if returned_session else 0.5)
- `satisfaction_score` = user_rating / 5 (or inferred from engagement if no explicit rating)

### Arm Selection Strategy

**Thompson Sampling / Posterior Sampling:**

```python
def get_next_arm():
    # Sample from posterior distribution of each arm
    samples = {}
    for arm in all_36_arms:
        samples[arm] = sample_from_posterior(arm)

    # With 80% probability, select highest sampled value
    if random() < 0.8:
        return argmax(samples)
    else:
        # 20% probability: random exploration
        return random_arm()
```

This balances **exploitation** (using arms with high posterior mean) and **exploration** (discovering hidden patterns).

### Posterior Updates

**Frequency:** Every N completed phrases (~10) or every 5 minutes (whichever comes first)

**Process:**
1. Query DuckDB for all trial outcomes since last update
2. Aggregate metrics per-arm (average objective across users)
3. Call Ax `optimize()` with new trial data
4. Update internal experiment state
5. Cache arm probabilities for next `get_next_arm()` call

**Handling sparse data:**
- Use weak priors (uniform distribution initially)
- Minimum trial count per arm before Bayesian update (e.g., 5 trials)
- Early phase: more exploration, less exploitation

---

## Error Handling and Edge Cases

### Graceful Degradation

| Failure Mode | Behavior |
|---|---|
| Ax optimization fails | Fallback to uniform random arm selection; log error; app continues |
| DuckDB unavailable | In-memory variant state only; metrics lost on restart; app continues |
| No user rating provided | Infer satisfaction from engagement metrics |
| Insufficient data for posterior | Use uniform prior, continue exploring |

### Edge Cases

**New user with no history:**
- Random arm assignment
- Log first phrase attempt
- Bayesian updates begin after 5-10 trials

**Returning user across sessions:**
- Retrieved from DuckDB via user_id
- Assigned to current best arm (exploitation)
- Contributes new data to posterior

**User device switching:**
- Treated as new user (no cross-device tracking without explicit login)
- Creates multiple user_ids per physical user (acceptable for now)

**HF Spaces ephemeral storage:**
- DuckDB file persists during Space uptime
- Lost on Space restart (acceptable limitation)
- Solution: Optional backup to Hugging Face Hub (out of scope)

**Variant lock-in:**
- If user stuck on low-performing variant, adaptive switch can suggest change
- User can accept/decline (respects user agency)
- Override mechanism for admin testing

### Data Privacy

- No PII collected (session_id only)
- No external API calls
- DuckDB stored locally on Space
- GDPR-compliant (no personal data tracking)

---

## Testing Strategy

### Unit Tests
- `test_bayesian_optimizer.py`: Test arm selection, posterior updates, edge cases
- `test_ab_test_logging.py`: Test metric logging, schema validation, exports
- `test_variant_manager.py`: Test assignment logic, state persistence

### Integration Tests
- Simulate 100 user sessions with various curriculum variants
- Verify metrics logged correctly
- Verify posterior converges to best arm over time
- Verify Gradio admin panel displays data correctly

### Manual Testing
1. Run locally with `python app.py`
2. Complete 2-3 sessions across different modes
3. Verify DuckDB file created and populated
4. Check admin panel shows arm probabilities
5. Export metrics and spot-check values

### Performance Testing
- Verify Ax optimization runs in < 1 second
- Verify phrase fetching adds < 100ms latency
- Verify DuckDB queries complete in < 100ms

---

## Implementation Phases

### Phase 1: Core infrastructure
- Create `bayesian_optimizer.py`, `ab_test_logging.py`, `variant_manager.py`
- Implement DuckDB schema and logging
- Unit tests for each module

### Phase 2: Gradio integration
- Integrate into `app.py` startup and learning mode functions
- Add admin panel with results visualization
- Integration tests

### Phase 3: Refinement
- Performance tuning (Ax optimization speed)
- Handle edge cases (sparse data, user switching)
- Documentation and deployment

---

## Success Criteria

1. ✅ A/B test framework deployed without breaking existing functionality
2. ✅ Metrics logged for all user phrases (engagement, retention, satisfaction)
3. ✅ Bayesian optimization converges to best variant within 100-200 user sessions
4. ✅ Admin panel shows live arm probabilities and convergence
5. ✅ Zero PII collection, fully privacy-compliant
6. ✅ Graceful degradation if Ax fails (app continues with random selection)

---

## Future Enhancements (Out of Scope)

- Multi-user backend for cross-device variant continuity
- Real-time convergence visualization (live updating chart)
- Contextual bandits (personalize variants by user language level)
- Sequential testing with early stopping (stop low-performing arms faster)
- Export results to Hugging Face Hub for persistence across Space restarts
