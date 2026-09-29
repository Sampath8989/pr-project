# All measured accuracy results

| Output / method | Evaluation data | Correct / total | Result | What this measures |
| --- | --- | ---: | ---: | --- |
| Position-only baseline | 4-agent validation | 5 / 13 | 38.5% | Top-choice agent accuracy |
| First-error keyword baseline | 4-agent validation | 4 / 13 | 30.8% | Top-choice agent accuracy |
| Regularized logistic baseline | 4-agent validation | 7 / 13 | 53.8% | Top-choice agent accuracy |
| Bayesian network, before calibration | 4-agent validation | 6 / 13 | 46.2% | Top-choice agent accuracy; validation guided feature selection |
| Position-only baseline | 4-agent test | 4 / 13 | 30.8% | Top-choice agent accuracy |
| First-error keyword baseline | 4-agent test | 3 / 13 | 23.1% | Top-choice agent accuracy |
| Regularized logistic baseline | 4-agent test | 4 / 13 | 30.8% | Top-choice agent accuracy |
| Bayesian network, before calibration | 4-agent test | 5 / 13 | 38.5% | Top-choice agent accuracy |
| Bayesian network, calibrated **current default** | 4-agent test | **5 / 13** | **38.5%** | Top-choice agent accuracy; calibration changes probabilities, not ranking |
| Bayesian network, complete logs | 4-agent test | 5 / 13 | 38.5% | Top-choice accuracy in the missing-log study |
| Bayesian network, third agent's log missing | 4-agent test | 5 / 13 | 38.5% | Top-choice accuracy after masking agent index 2 |
| Bayesian network, one random agent log missing | 4-agent test | 6 / 13 | 46.2% | Top-choice accuracy for one fixed random mask per trace; missing data does not generally help |
| Bayesian network, two random agent logs missing | 4-agent test | 4 / 13 | 30.8% | Top-choice accuracy for two fixed random masks per trace |
| Bayesian network | 2-agent test | 3 / 3 | 100.0% | Exploratory top-choice accuracy; very small group |
| Bayesian network | 3-agent test | 3 / 7 | 42.9% | Exploratory top-choice accuracy; small group |
| Bayesian network | 4-agent test | 5 / 13 | 38.5% | Primary top-choice accuracy, repeated here for agent-count comparison |
| Bayesian network two-agent shortlist | 4-agent test | 10 / 13 | 76.9% | True agent appears in the top two; **not** top-choice accuracy |
| HMM early-warning model | Eligible 4-agent test traces | 2 / 10 | 20.0% | Correct agent warned before the labeled mistake; early recall |
| Keyword early-warning baseline | Eligible 4-agent test traces | 1 / 10 | 10.0% | Correct agent warned before the labeled mistake; early recall |
| HMM wrong-agent warnings | 4-agent temporal test traces | 11 / 12 | 91.7% | Undesirable false-warning rate; **lower is better** |
| Keyword wrong-agent warnings | 4-agent temporal test traces | 6 / 12 | 50.0% | Undesirable false-warning rate; **lower is better** |
| Bayesian network | Five question-group folds, 4-agent traces | 40 / 91 | 44.0% | Out-of-fold top-choice accuracy |
| Text ranker trained across agent counts | Five question-group folds, 4-agent traces | 33 / 91 | 36.3% | Out-of-fold top-choice accuracy; below BN |
| Text ranker trained across agent counts | Previously inspected 4-agent test | 7 / 13 | 53.8% | Exploratory top-choice result; **not a fresh holdout improvement** |
| Lexical and handoff ranker | Previously inspected 4-agent test | 6 / 13 | 46.2% | Exploratory top-choice result; worse probability scores than calibrated BN |
| Ranker with filtered StepFinder generated traces | Previously inspected 4-agent test | 3 / 13 | 23.1% | Exploratory top-choice result |
| Semantic-embedding ranker with augmented traces | Previously inspected 4-agent validation | 9 / 13 | 69.2% | Exploratory selection result; did not generalize |
| Semantic-embedding ranker with augmented traces | Previously inspected 4-agent test | 3 / 13 | 23.1% | Exploratory top-choice result |
| BN after excluding task-instruction messages | Previously inspected 4-agent test | 5 / 13 | 38.5% | Exploratory top-choice result; no improvement |
| Archived external-data step model (L2=1.0), agent | All algorithm-generated Who&When traces | 63 / 126 | 50.0% | Strict exact agent accuracy; archived prior setting |
| Archived external-data step model (L2=1.0), step | All algorithm-generated Who&When traces | 34 / 126 | 27.0% | Strict exact raw-step accuracy; archived prior setting |
| Archived external-data step model (L2=1.0), agent | All handcrafted Who&When traces | 36 / 58 | 62.1% | Strict exact agent accuracy; archived prior setting |
| Archived external-data step model (L2=1.0), step | All handcrafted Who&When traces | 9 / 58 | 15.5% | Strict exact raw-step accuracy; archived prior setting |
| **Tuned external-data step model (L2=0.45), agent** | All algorithm-generated Who&When traces | **62 / 126** | **49.2%** | Strict exact agent accuracy; 95% bootstrap CI 40.5–57.9% |
| **Tuned external-data step model (L2=0.45), step** | All algorithm-generated Who&When traces | **36 / 126** | **28.6%** | Strict exact raw-step accuracy; 95% bootstrap CI 20.6–36.5% |
| **Tuned external-data step model (L2=0.45), agent** | All handcrafted Who&When traces | **37 / 58** | **63.8%** | Strict exact agent accuracy; 95% bootstrap CI 51.7–75.9% |
| **Tuned external-data step model (L2=0.45), step** | All handcrafted Who&When traces | **10 / 58** | **17.2%** | Strict exact raw-step accuracy; 95% bootstrap CI 8.6–27.6% |
| Bayesian network | Nested question-group folds, 4-agent traces | 40 / 91 | 44.0% | Paired baseline for the hybrid |
| **Hybrid BN + tuned step model** | Nested question-group folds, 4-agent traces | **48 / 91** | **52.7%** | Improved over BN; paired p = 0.077, exploratory |
| Bayesian network refit on train + validation | Previously inspected 4-agent test | 5 / 13 | 38.5% | Paired baseline for the hybrid on the same training cases |
| **Hybrid BN + step model** | Previously inspected 4-agent test | **6 / 13** | **46.2%** | Improved top-choice accuracy; test reused during prior research |

Sources: [baseline metrics](artifacts/phase2_evidence_baselines/baselines.json), [BN validation](artifacts/phase3_bayesian_inference/validation_predictions.jsonl), [missing-log metrics](artifacts/phase4_missing_evidence/metrics.json), [calibrated test metrics](artifacts/phase5_calibration_evaluation/calibration.json), [agent-count metrics](artifacts/phase5_calibration_evaluation/exploratory_agent_count_metrics.json), [temporal metrics](artifacts/phase6_temporal_hmm/metrics.json), [top-two metrics](artifacts/phase9_evidence_triage/metrics.json), [group-fold accuracy study](accuracy_study/results.json), [external-data step model](benchmark_step/metrics.json), [hybrid study](hybrid_upgrade/metrics.json), and [exploratory experiment audit](RESEARCH_UPGRADE.md). The 13-case test has been reused during later experiments, so those exploratory scores are not independent confirmation.
