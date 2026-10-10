"""Build the October review guide from preserved measurements and current code."""
from pathlib import Path
from xml.sax.saxutils import escape
import ast
import json
import hashlib
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'output'/'pdf'
DEST.mkdir(parents=True,exist_ok=True)
OUT=DEST/'AgentDiag_v2_Updated_Project_Review_Guide.pdf'
NAVY=colors.HexColor('#12233F'); TEAL=colors.HexColor('#0891B2'); BLUE=colors.HexColor('#246BFD')
INK=colors.HexColor('#23324B'); PALE=colors.HexColor('#F1F5FA'); CORAL=colors.HexColor('#B45365')
pdfmetrics.registerFont(TTFont('Review','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ReviewBold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Review',normal='Review',bold='ReviewBold',italic='Review',boldItalic='ReviewBold')
styles={
 'body':ParagraphStyle('body',fontName='Review',fontSize=10,leading=15,textColor=INK,spaceAfter=8),
 'h1':ParagraphStyle('h1',fontName='ReviewBold',fontSize=23,leading=28,textColor=NAVY,spaceAfter=15,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='ReviewBold',fontSize=13,leading=18,textColor=TEAL,spaceBefore=8,spaceAfter=6,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Review',fontSize=8,leading=11,textColor=INK),
 'head':ParagraphStyle('head',fontName='ReviewBold',fontSize=8,leading=11,textColor=colors.white),
 'small':ParagraphStyle('small',fontName='Review',fontSize=8,leading=11,textColor=INK,spaceAfter=6),
 'cover':ParagraphStyle('cover',fontName='ReviewBold',fontSize=35,leading=42,textColor=NAVY,spaceAfter=18),
}
story=[]; sections=[]
def p(text,kind='body'): return Paragraph(text,styles[kind])
def add(text): story.append(p(text))
def bullets(items):
    for item in items: story.append(p('<b>-</b> '+item))
def heading(title): story.append(p(title,'h2'))
def page(title,sub=''):
    if story: story.append(PageBreak())
    sections.append(title);story.append(p(title,'h1'))
    if sub: add(sub)
def table(headers,rows,widths=None):
    widths=widths or [500/len(headers)]*len(headers)
    data=[[p(escape(str(c)),'head') for c in headers]]
    data.extend([[p(escape(str(c)),'cell') for c in row] for row in rows])
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#D8E2EF')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,10)])
def graph(file,caption):
    story.append(Image(str(ROOT/'results_graphs'/file),width=500,height=250))
    story.append(p(caption,'small'))
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
m=load('contextual_upgrade/metrics.json')
sel=load('contextual_upgrade/selection.json')
manifest=load('results_graphs/source_manifest.json')

story.extend([Spacer(1,45),p('AGENTDIAG<br/>PROJECT REVIEW GUIDE','cover'),p('Updated edition | 10 October 2026','h2'),p('A simple-English explanation of the project, every development stage, the older and newer models, measured results, evidence, files and likely review questions.'),Spacer(1,25)])
table(['184','57.9%','33.3%'],[['Saved evaluation conversations','Updated Algorithm-Generated agent accuracy','Updated Algorithm-Generated exact step accuracy']])
add('<b>Project:</b> Failure diagnosis in teams of AI agents.<br/><b>Subject connection:</b> Probabilistic reasoning and uncertainty.<br/><b>Scope:</b> Local analysis of saved text conversations.')
add('This edition updates the earlier 18-page review guide. It preserves the actual recorded results and includes the contextual architecture upgrade and the result graph pack.')
add('Code reference: local branch feature/contextual-attribution. This guide documents local work; the GitHub main branch has not received these latest changes.')
page('How to use this guide')
bullets(['For a short presentation, read the overview, version comparison and results pages first.','For technical questions, use the architecture, probabilistic reasoning and file explanations.','For evidence, use the results table, prediction examples, graph appendix and verification pages.','For a viva, practise the questions and answers near the end.','The graph appendix contains all 26 exported figures. Each chart names its evaluation setting.'])
table(['Part','What it explains'],[['Project and data','What problem we solve, what the input means and how testing works.'],['Models and versions','Original probabilistic pipeline, older step model and latest contextual model.'],['Changes and results','What was changed, what improved and what still needs work.'],['Evidence and files','How to verify the scores and what each source file or output family does.'],['Review preparation','Plain answers to common faculty questions.'],['Appendices','All 26 graphs, glossary, source links and complete file inventory.']],[120,380])
heading('Version names used in this document')
add('The folder name agentdiag_v2 refers to the fresh project implementation. Inside that folder there are several model revisions. To avoid confusion, this guide calls the immediately previous L2=0.45 step ranker <b>V1 / previous ranker</b> and the latest contextual candidate <b>V2 / contextual model</b>. These are comparison labels; the earlier Bayesian pipeline remains a separate component.')
page('1. The project in one minute')
add('Imagine a team of AI workers solving a task. One reads data, another calculates, and another writes the answer. If the final answer is wrong, someone must investigate where the team went wrong. AgentDiag reads the saved conversation and suggests which worker and which message deserve attention.')
heading('The two main questions')
bullets(['<b>Who?</b> Which agent is recorded as responsible for the failure?','<b>When?</b> Which exact message is recorded as the decisive mistake?'])
heading('What comes in and what goes out')
table(['Input','Output'],[['A saved JSON conversation: task, speaker names and messages.','A predicted agent, a predicted message index, and probability scores.'],['An evaluation record also has the correct labels.','Evaluation code compares predictions with the labels after prediction.'],['The probabilistic pipeline extracts evidence clues.','A review packet gives ranked candidates, excerpts, a shortlist and sensitivity checks.']],[250,250])
heading('A useful one-minute explanation')
add('"My project helps investigate failed AI-team conversations. It learns patterns from labelled examples and ranks possible responsible agents and messages. The new contextual model improved the generated-conversation scores. A separate Bayesian model explains uncertainty and handles missing evidence. We verify the predictions against saved labels and provide files and graphs for review."')
page('2. What we actually tested')
add('We tested diagnosis on saved conversations. We did not download and launch every original agent team, and we did not run live counterfactual replay. The local model reads an existing record and predicts its failure labels.')
table(['Dataset / use','Size','Why it is used'],[['Original Who&When evaluation','184 traces: 126 Algorithm-Generated + 58 Hand-Crafted','Evaluate agent and exact-step attribution.'],['External StepFinder training source','2,004 filtered traces from 58 questions','Train the previous and contextual models without using original target labels as training answers.'],['Original probabilistic pipeline','134 train / 27 validation / 23 test traces','Build and evaluate the Bayesian components.'],['Primary four-agent probabilistic split','65 train / 13 validation / 13 test','Small, comparable Bayesian task.'],['External grouped development folds','Five question-group folds','Select model settings and probability blends.']],[150,160,190])
heading('What the dataset names mean')
add('<b>Algorithm-Generated</b> and <b>Hand-Crafted</b> describe how the agent systems were constructed. They do not mean that we manually invented our accuracy numbers. Both groups contain saved conversations with reference answers.')
heading('Prediction and labels')
add('At prediction time, parse_unlabeled keeps observed messages, agent information and the task question. It ignores mistake_agent, mistake_step, ground_truth and is_correct. Correct labels are used by training and evaluation code, not provided as answers to the predictor.')
page('3. How one conversation is processed')
table(['Step','Action','Simple explanation'],[['1','Read the trace','Load the task, agents and messages from JSON.'],['2','Prepare observed information','Keep speaker names and original message indices. Hide answer fields from prediction.'],['3','Extract clues','Represent text, roles, positions, corrections, uncertainty and nearby messages.'],['4','Predict who','Assign a probability to each candidate agent.'],['5','Predict when','Assign a probability to each candidate message.'],['6','Return the result','Choose the highest-scoring agent and exact message.'],['7','Evaluate later','Compare those predictions with the stored reference labels.']],[35,135,330])
heading('Example, in ordinary language')
add('If a worker makes an incorrect calculation and later workers repeat it, the model should identify the original error rather than only the final wrong answer. That is difficult: the visible conversation can contain several mistakes, corrections and misleading explanations.')
heading('Why later messages are useful')
add('The diagnosis happens after a run, so it can inspect later messages for evidence that an earlier answer was rejected or corrected. The temporal neural candidate reads both directions. This is suitable for post-run diagnosis; it is not a real-time warning model.')
page('4. Connection to probabilistic reasoning')
heading('Bayesian reasoning')
add('The Bayesian network represents one possible responsible agent as a categorical variable. Evidence clues influence the probability of each agent. In simple notation: P(agent | observed clues) is proportional to P(observed clues | agent) multiplied by P(agent). The probabilities are normalized so they add up to one.')
heading('Variable Elimination')
add('Variable Elimination calculates the required answer without keeping every possible combination in memory. It combines probability tables and sums over variables that are not needed. The project implements this in src/bayes.py and checks sampled results against full enumeration.')
heading('Missing evidence')
add('If some messages are missing, the model can sum over the possible values of the missing clues. Missing is not automatically interpreted as "no error". This is one reason Bayesian reasoning fits the subject.')
heading('Calibration')
add('A model may say it is 80% confident but be correct much less often. Calibration adjusts the confidence values. On the 13-case Bayesian test, calibration improved probability error scores while the top-choice result stayed 5/13. The latest contextual probabilities have not been calibrated.')
heading('Important distinction')
add('The latest 57.9% / 33.3% benchmark comes from the contextual ensemble, not from the Bayesian network. Both are part of the project, but their measured results and roles must be explained separately.')
page('5. Original pipeline: every phase')
phases=[('0','Protocol','Audit 184 traces and freeze question-group splits.','phase0_protocol'),('1','Parsing','Read messages and labels; retain raw indices; flag label mismatches.','phase1_parsing'),('2','Evidence and baselines','Extract clues and measure simple and logistic predictors.','phase2_evidence_baselines'),('3','Bayesian inference','Fit probability tables and calculate agent probabilities.','phase3_bayesian_inference'),('4','Missing evidence','Mask messages and measure changes in predictions.','phase4_missing_evidence'),('5','Calibration and evaluation','Fit confidence adjustment on validation and report held-out metrics.','phase5_calibration_evaluation'),('6','Temporal HMM','Attempt earlier warnings and count wrong-agent warnings.','phase6_temporal_hmm'),('7','Multimodal scope audit','Check whether the supplied data supports an image/audio/video claim.','phase7_multimodal_scope'),('8','Final report','Collect results, limits, verification and a review presentation.','phase8_final_report'),('9','Evidence triage','Create shortlists, source excerpts and model-sensitivity review packets.','phase9_evidence_triage')]
table(['Phase','Purpose','What was done'],[[a,b,c] for a,b,c,d in phases],[40,130,330])
add('Each phase has a report under artifacts/&lt;phase folder&gt;/. Machine-readable JSON or JSONL files sit beside it. The phase manifest lists the expected output files. A completed phase means the implementation ran; it does not mean its accuracy is high.')
page('6. Development history and earlier versions')
table(['Stage','Changes','Measured conclusion'],[['Fresh AgentDiag v2 pipeline','Separate folder; parsing, Bayesian inference, missing evidence, calibration, HMM and reports.','Frozen four-agent Bayesian test: 5/13 (38.5%).'],['Evidence-review extension','Added a two-agent shortlist, message excerpts and sensitivity checks.','Shortlist includes actual agent in 10/13 (76.9%).'],['Archived external step model, L2=1.0','Trained a compact message ranker on filtered external data.','Algorithm agent 63/126 (50.0%), step 34/126 (27.0%).'],['Tuned external step model, L2=0.45','Selected a regularization setting using external grouped validation.','Algorithm agent 62/126 (49.2%), step 36/126 (28.6%); Hand-Crafted 37/58 and 10/58.'],['Probabilistic hybrid','Combined Bayesian and external ranker probabilities.','Grouped four-agent folds: 48/91 vs Bayesian 40/91; p=0.077.'],['Windows migration audit','Restored files from GitHub and checked 184 data records; repaired portable paths.','Data agreed with the local upstream copy; recorded checks passed.'],['Further local attempts','Agent-aware step scoring, domain adaptation and subset specialists.','Validation retained the previous model; no new gain claimed.'],['Contextual upgrade','Question context, lexical context, semantic encoder, temporal candidates and separate heads.','Algorithm agent 73/126 and step 42/126. Hand-Crafted step dropped to 8/58.'],['Result evidence export','26 PNG charts, 184-row prediction CSV, metric CSVs and source hashes.','Graph values and prediction counts checked.']],[110,215,175])
page('7. V1: the previous step ranker')
add('The previous model is a small linear ranker. It turns words, roles and message positions into numeric features, then learns a score for each possible mistake message. It is cheap to train and runs locally.')
bullets(['Uses message text and the following message as evidence.','Uses agent-role and position features.','Learns exact message localization from labelled external traces.','Obtains agent probabilities by combining step probabilities for messages belonging to each agent.','L2 regularization discourages very large weights. The tuned setting is 0.45.'])
heading('Why it can struggle')
add('Short text clues may not show whether an answer is relevant to the original task. A model can confuse the first obvious error word with the real initial mistake. Inferring the agent only from step scores also links the two errors: an incorrect step can produce an incorrect agent.')
heading('Recorded previous-model results')
table(['Subset','Agent','Exact step'],[['Algorithm-Generated','62/126 (49.2%)','36/126 (28.6%)'],['Hand-Crafted','37/58 (63.8%)','10/58 (17.2%)']],[180,160,160])
page('8. V2: the contextual architecture')
heading('A. Contextual lexical model')
add('This model reads individual words and pairs of words. It also includes preceding and subsequent messages, authored content, question overlap, role clues and positions. Separate classifiers learn the agent answer and the exact step answer directly.')
heading('B. Semantic model')
add('Qwen3-Embedding-0.6B converts text into vectors that represent meaning. Its pretrained weights remain frozen. A smaller neural model learns from those vectors, task interactions, role information and observed numeric clues. A bidirectional GRU tracks the message sequence; attention helps it focus on useful context. Agent mean/max pooling collects evidence for each speaker.')
heading('C. Separate training targets')
add('An agent head learns which agent is correct. A step head learns the exact message. The experiments compare exact losses and an optional small distance-based step loss. Training tries to assign higher probabilities to the recorded correct labels.')
heading('D. The selected combination')
table(['Output','Final probability combination'],[['Agent','50% previous ranker + 25% contextual lexical + 25% temporal neural ensemble.'],['Exact step','25% previous ranker + 75% contextual lexical.'],['Neural ensemble','Two fixed seeds, 13 and 29; temporal_exact configuration trained for 16 epochs.']],[120,380])
add('The semantic model complements agent prediction. The selected step output uses lexical context and the previous ranker. A more complicated model was not automatically treated as better.')
page('9. Complete change list: what changed and why')
table(['Change','Reason','Evidence / implementation'],[['Keep the task question in label-free parsing','Determine whether message content fits the actual task.','src/data.py; answer fields still ignored.'],['Add words in pairs and surrounding messages','Recognize corrections and dependencies beyond a single message.','src/contextual_sparse.py.'],['Directly train an agent classifier','Reduce reliance on deriving agents solely from step scores.','Separate lexical and neural agent heads.'],['Add frozen semantic embeddings','Represent meaning as well as keyword overlap.','src/semantic_encoder.py; pinned Qwen encoder.'],['Add task interaction and temporal modeling','Use the task and conversation order to distinguish likely errors.','src/contextual_ranker.py; GRU and attention.'],['Use separate agent and step blends','The best source of agent evidence may differ from the best source of step evidence.','selection.json, selected using external grouped predictions.'],['Compare settings systematically','Measure architectures instead of choosing one by appearance.','Three neural configurations, two seeds, three epoch checkpoints; 30 fold fits.'],['Cache embeddings by content','Avoid rerunning the large encoder for unchanged text.','Content-addressed cache and packed vectors.'],['Optimize numeric preprocessing','Avoid repeated feature extraction.','Matrices unchanged on checked traces; about 30 times faster in the recorded check.'],['Save checkpoints and selection hashes','Make chosen model outputs recoverable and auditable.','Saved model files, source hashes, CV predictions and verification.json.'],['Add graph and CSV evidence','Make report numbers easy to inspect.','results_graphs; counts rechecked from all 184 predictions.']],[145,205,150])
page('10. Training, selection and local resources')
bullets(['Training source: StepFinder repository commit 48d71c7090adca3667a31b043a78484f407b4a2f.','2,004 retained traces from 58 questions; the source data remained unchanged.','Overlap filtering: token Jaccard below 0.85, then BGE-small maximum cosine similarity to original questions below 0.75. This reduces obvious overlap; it cannot prove all semantic leakage is absent.','Five folds keep related traces of a question together.','Three neural configurations: temporal_exact, temporal_distance and attention_exact.','Two seeds, 13 and 29; saved epochs 8, 16 and 24. Thirty neural fold fits.','Lexical settings 0.5 and 2.0, plus probability mixtures.','Agent and step settings selected on external grouped predictions before scoring this round on original labels.'])
heading('What ran on the local machine')
add('The machine has an NVIDIA RTX 2050 with 4 GB of GPU memory. Qwen3-Embedding-0.6B ran locally; the smaller predictors were trained locally. PyTorch 2.8.0 with CUDA 12.6 and Transformers 4.57.6 were used. No model API or cloud agent execution was used for this upgrade.')
heading('Input budget and model identity')
add('Encoder revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3. Text vectors have 256 dimensions. Messages use a 384-token budget; longer inputs retain the first 288 and last 96 tokens. These limits save memory but can omit important information.')
page('11. Original version-comparison results')
table(['Subset','Version','Agent correct','Exact step correct','Both correct'],[[g,'V1 / previous',f"{m['old'][g]['correct_agent']}/{m['old'][g]['traces']} ({100*m['old'][g]['agent_accuracy']:.1f}%)",f"{m['old'][g]['correct_step']}/{m['old'][g]['traces']} ({100*m['old'][g]['step_accuracy']:.1f}%)",f"{m['old'][g]['correct_both']}/{m['old'][g]['traces']} ({100*m['old'][g]['joint_accuracy']:.1f}%)"] for g in ['Algorithm-Generated','Hand-Crafted']]+[[g,'V2 / contextual',f"{m['new'][g]['correct_agent']}/{m['new'][g]['traces']} ({100*m['new'][g]['agent_accuracy']:.1f}%)",f"{m['new'][g]['correct_step']}/{m['new'][g]['traces']} ({100*m['new'][g]['step_accuracy']:.1f}%)",f"{m['new'][g]['correct_both']}/{m['new'][g]['traces']} ({100*m['new'][g]['joint_accuracy']:.1f}%)"] for g in ['Algorithm-Generated','Hand-Crafted']],[110,90,100,100,100])
heading('How to calculate the score')
add('Accuracy = correct predictions / number of evaluated conversations x 100. Agent accuracy requires the exact canonical agent. Step accuracy requires the exact raw message index. Joint accuracy requires both to be correct on the same trace.')
heading('What improved')
bullets(['Algorithm-Generated agent: 62 to 73 correct, a net gain of 11 and +8.7 percentage points.','Algorithm-Generated exact step: 36 to 42 correct, a net gain of 6 and +4.8 percentage points.','Algorithm-Generated both correct: 33 to 36.','Hand-Crafted agents: unchanged at 37/58.','Hand-Crafted exact steps: 10 to 8; a regression of 3.4 percentage points.'])
add('The retained older predictor remains preferable for Hand-Crafted steps. This recommendation is different from claiming that the new contextual model improved that subset. No automatic routing or independently validated selector is claimed.')
page('12. Interpreting improvements carefully')
table(['Measure','Wrong to correct','Correct to wrong','Net change','Paired p-value'],[['Algorithm agent','16','5','+11','0.0266'],['Algorithm step','11','5','+6','0.2101'],['Hand-Crafted agent','3','3','0','1.0000'],['Hand-Crafted step','1','3','-2','0.6250']],[130,95,95,85,95])
add('The paired test compares the same conversations under the previous and updated models. The Algorithm-Generated agent gain is stronger evidence than the exact-step gain in this comparison. However, benchmark reuse and model development still limit the conclusion; these p-values do not establish an independent paper win.')
heading('Uncertainty')
add('Saved 95% trace-bootstrap intervals for updated Algorithm-Generated accuracy are 49.2%-65.9% for agents and 25.4%-41.3% for steps. These intervals are wide and do not correct for development-set reuse or all dependence between related questions.')
heading('Requested goals')
add('The requested 60% agent and 35% exact-step targets are not met. With 126 examples, reaching them requires at least 76 correct agents and 45 correct steps. The current model has 73 and 42, so it needs three additional correct predictions for each metric.')
heading('Development versus original data')
add('Selected external Algorithm-Generated development accuracy was 781/989 agents (79.0%) and 681/989 steps (68.9%). Original-corpus scores are lower. This gap suggests that patterns learned from external data do not transfer fully to original conversations. Development scores were used for selection, so they are not untouched final-test claims.')
page('13. Comparison with baseline papers')
table(['Method','Algorithm agent','Algorithm step','Hand-Crafted agent','Hand-Crafted step'],[['V1 / previous local','49.2%','28.6%','63.8%','17.2%'],['V2 / contextual local','57.9%','33.3%','63.8%','13.8%'],['Who&When GPT-4o, reported','51.12%','13.53%','53.44%','3.51%'],['AgenTracer, reported','63.73%','37.30%','63.82%','20.68%']],[140,90,90,90,90])
add('Paper rows refer to the cited no-answer-supplied setting. They are published numbers, not models rerun by this project on a fully matched protocol. Differences in answer parsing, exact-step scoring, data selection and development exposure matter.')
heading('What the papers did')
bullets(['Who&When: used a language-model judge with the question and saved trajectory. It compared reading the whole log, increasing prefixes and binary search. Agent systems were run to construct the benchmark; diagnosis methods then analysed saved records.','AgenTracer: generated training trajectories through replay and injected faults, then used a Qwen3-8B model with GRPO training and agent, step-distance and format rewards. Its cited training used eight H100 80 GB GPUs.','StepFinder: its authors used Qwen embeddings, a bidirectional LSTM, agent interaction attention and temporal features. Our local candidate uses a GRU and separate supervised heads and is not a complete reproduction.'])
heading('Fair conclusion')
add('Our updated Algorithm-Generated percentages are numerically above the cited older Who&When GPT-4o row and below AgenTracer. A matched independent evaluation is needed before claiming benchmark superiority. Matching a percentage alone does not demonstrate equivalent performance.')
page('14. Why accuracy is still limited')
bullets(['<b>Different data distributions:</b> external failure patterns differ from original logs. The validation-to-original gap supports this concern.','<b>Model capacity and compute:</b> a frozen 0.6B encoder plus compact predictors cannot be assumed equivalent to an 8B reasoning model with dedicated reinforcement learning.','<b>No replay evidence:</b> we observe logs, but do not rerun the agent after changing a suspected action.','<b>Ambiguous mistakes:</b> the first wrong statement and the decisive failure label may differ.','<b>Exact-step scoring:</b> a prediction one message away is still wrong.','<b>Long conversations:</b> the token budget can omit important middle text.','<b>Small subsets:</b> two cases change the Hand-Crafted score by 3.4 percentage points.','<b>Confidence:</b> contextual probabilities are not yet calibrated.'])
heading('What to try next')
add('Use more representative independent labelled failures; improve long-message context; test a properly nested residual correction model; calibrate probabilities on separate validation data; freeze settings and evaluate on unseen traces with the same scorer. More experiments may help, but no particular accuracy or faculty mark is guaranteed.')
page('15. Other implemented results')
table(['Component / setting','Measured result','Interpretation'],[['Bayesian frozen four-agent test','5/13 = 38.5%','A working probability engine with limited attribution accuracy.'],['Bayesian shortlist','10/13 = 76.9%','Actual agent appears in the top two; not top-one accuracy.'],['Bayesian / hybrid grouped folds','40/91 = 44.0% / 48/91 = 52.7%','Hybrid gain is exploratory; paired p=0.077.'],['Bayesian / hybrid inspected test','5/13 / 6/13','Small and previously inspected comparison.'],['BN complete / one random missing / two missing','5/13 / 6/13 / 4/13','Small-sample masking experiment; missing data does not generally improve accuracy.'],['HMM correct early warnings','2/10 = 20.0%','Early-warning reliability is weak.'],['HMM wrong-agent warnings','11/12 = 91.7%','High undesirable warning rate.'],['Keyword early baseline','1/10 early; 6/12 wrong-agent warnings','Separate early-warning comparison.']],[190,150,160])
heading('Calibration results')
add('Bayesian Brier error improved from 0.814 to 0.689; log loss from 1.424 to 1.238; expected calibration error from 0.438 to 0.146. Lower is better. These are probability-quality scores, not changes to the 5/13 top-choice accuracy.')
page('16. Experiments that were not adopted')
table(['Experiment','Recorded outcome','Decision'],[['Earlier semantic / lexical rankers','Some settings looked strong on validation but scored poorly on inspected tests.','Keep the failed attempts as evidence; do not claim an improvement.'],['Agent-aware step consistency','External validation selected an unchanged setting.','No gain claimed.'],['Domain adaptation','Validation retained the existing external model.','No gain claimed.'],['Subset specialists','External grouped selection rejected the specialist.','Keep existing model.'],['New soft agreement','Original Algorithm agents fell from 73/126 to 70/126; exact steps stayed 42/126; joint rose 36 to 40.','Saved as experiment, not applied by predict_contextual.'],['Soft agreement on Hand-Crafted','Step count rose 8 to 9, still below previous 10.','Does not resolve the regression.']],[130,245,125])
add('The agreement experiment improved its external development score, but the original agent result fell. This is a useful lesson: validation gains can fail to transfer. Saving unsuccessful attempts improves reproducibility and prevents selective reporting.')
page('17. Evidence: how results can be checked')
table(['Evidence file','What it supports'],[['contextual_upgrade/benchmark_predictions.jsonl','Every current prediction, label and probability vector.'],['contextual_upgrade/metrics.json','Counts, accuracies, intervals and paired gains/losses.'],['contextual_upgrade/selection.json','Chosen settings, external validation scores and source hashes.'],['contextual_upgrade/verification.json','Recorded reproduction of all 184 predictions and validity checks.'],['contextual_upgrade/model_*.pt and model_sparse_0.5.json','Saved trained predictors used to reproduce outputs.'],['results_graphs/prediction_evidence.csv','184 rows of actual and predicted agent/step values.'],['results_graphs/graph_values.csv','Numeric values used in bar charts.'],['results_graphs/source_manifest.json','SHA-256 hashes and figure captions.'],['results_graphs/saved_phase_metrics.csv','Scalar metrics exported from saved phase reports.']],[245,255])
heading('Verification already recorded for the upgrade')
bullets(['16 automated tests passed in the completed accuracy run.','17 original artifact checks and 13 earlier upgrade checks passed.','All 184 saved contextual predictions were reproduced from label-free inputs.','Exact counts, valid agents and raw steps, finite normalized probabilities and selection hashes were checked.','All 26 PNG files were checked as valid images; prediction counts were recomputed by the exporter.'])
add('These statements describe the saved verification run. Creating this PDF does not rerun model training or establish accuracy on new data. The PDF build separately checks that its tables agree with source metrics.')
page('18. Actual output examples')
rows=[json.loads(x) for x in (ROOT/'contextual_upgrade/benchmark_predictions.jsonl').read_text().splitlines() if x]
patterns=[('Both answers correct',lambda r:r['actual_agent']==r['predicted_agent'] and r['actual_step']==r['predicted_step']),('Correct agent, wrong step',lambda r:r['actual_agent']==r['predicted_agent'] and r['actual_step']!=r['predicted_step']),('Both answers wrong',lambda r:r['actual_agent']!=r['predicted_agent'] and r['actual_step']!=r['predicted_step'])]
for label,predicate in patterns:
    r=next(x for x in rows if x['subset']=='Algorithm-Generated' and predicate(x))
    heading(label)
    table(['Trace','Actual agent / step','Predicted agent / step'],[[r['id'],f"{r['actual_agent']} / {r['actual_step']}",f"{r['predicted_agent']} / {r['predicted_step']}"]],[160,170,170])
    add('Agent correct: '+str(r['actual_agent']==r['predicted_agent'])+'. Exact step correct: '+str(r['actual_step']==r['predicted_step'])+'.')
add('These rows are actual saved predictions, not invented demonstrations. To inspect the message content, open the matching original JSON under data/Who&amp;When/ and locate the raw message index. The model can be confidently wrong, so a human should still inspect the evidence.')
page('19. Main commands and a review demonstration')
table(['Purpose','Command from repository root'],[['Original pipeline','python agentdiag_v2/run_pipeline.py'],['Bayesian diagnosis','python agentdiag_v2/predict.py TRACE'],['Previous step ranker','python agentdiag_v2/predict_step.py TRACE'],['Probabilistic hybrid','python agentdiag_v2/predict_hybrid.py TRACE'],['Latest contextual prediction','python -m agentdiag_v2.predict_contextual TRACE --cache S:/pr/agentdiag-model-cache'],['Check contextual outputs','python -m agentdiag_v2.verify_contextual'],['Original artifact checks','python agentdiag_v2/verify.py'],['Earlier upgrade checks','python -m agentdiag_v2.verify_upgrade'],['Tests','python -m pytest agentdiag_v2/tests -q'],['Export graphs','python agentdiag_v2/export_result_graphs.py'],['Build this updated PDF','python agentdiag_v2/create_updated_review_pdf.py']],[170,330])
heading('How to demonstrate')
bullets(['Open one saved trace and explain the task and speakers.','Run the correct predictor and show its selected agent and step.','Show a Bayesian review packet if explaining probabilistic reasoning.','Open the prediction evidence CSV and locate the trace.','Show the version accuracy table and one error example.','Explain the current limitations and the next testing step.'])
add('Install normal dependencies from requirements.txt and optional neural dependencies from requirements-contextual.txt. Encoder files and external StepFinder training data live outside the repository. A first uncached prediction can need the pinned public encoder download; the verified local cache avoids that.')
page('20. File guide: pipeline and prediction entry points')
descriptions={
'run_pipeline.py':'Runs original phases and writes the phase reports and artifacts.',
'predict.py':'Diagnoses a trace with the Bayesian model and its evidence review packet.',
'predict_step.py':'Uses the older external-data step ranker to predict agent and raw step.',
'predict_hybrid.py':'Combines Bayesian and step-ranker probabilities for supported four-agent traces.',
'predict_contextual.py':'Runs the latest selected lexical / temporal ensemble on label-free input.',
'benchmark_step.py':'Trains/selects the external linear step model and scores original conversations.',
'hybrid_upgrade.py':'Fits and evaluates the earlier Bayesian-plus-step probability blend.',
'contextual_upgrade.py':'Runs sparse, encode, neural, select, train and evaluate stages for the new model.',
'contextual_consistency.py':'Tests agent/step agreement; saved experiment not adopted by the main predictor.',
'accuracy_study.py':'Produces the earlier grouped four-agent model comparison.',
'verify.py':'Checks original phase artifacts, counts, splits and outputs.',
'verify_upgrade.py':'Checks earlier step and hybrid benchmark outputs.',
'verify_contextual.py':'Recomputes metrics and reproduces every current checkpoint prediction.',
'export_result_graphs.py':'Generates 26 report graphs and supporting CSV/source evidence.',
'make_presentation.py':'Builds the original result-aligned review presentation.',
'create_project_review_pdf.py':'Original September review-guide generator; kept as historical source.',
'create_updated_review_pdf.py':'Builds this updated October PDF using preserved metrics and graph evidence.'}
table(['File','What it does'],list(descriptions.items()),[185,315])
page('21. File guide: source modules and tests')
modules={'src/data.py':'Loads records, parses labelled and label-free traces, groups questions and extracts observed clues.','src/bayes.py':'Probability tables, categorical Bayesian inference and Variable Elimination.','src/evaluation.py':'Accuracy, probability-quality metrics, confidence bins and evaluation utilities.','src/temporal.py':'Hidden Markov model and time-based warning logic.','src/triage.py':'Two-agent shortlist, source-message excerpts and sensitivity-based review packets.','src/ranker.py':'Earlier text/evidence ranking model experiments.','src/step_ranker.py':'Compact linear message-scoring model and agent marginal probabilities.','src/semantic_encoder.py':'Pinned local encoder, token budget, frozen text embeddings and reusable caches.','src/contextual_ranker.py':'Task-aware neural agent and step heads, sequence model, attention and checkpoints.','src/contextual_sparse.py':'Contextual word features and directly supervised linear agent and step models.','tests/test_pipeline.py':'Checks core pipeline behaviour including probability inference and parsing.','tests/test_step_ranker.py':'Checks external step-ranker behaviour.','tests/test_hybrid.py':'Checks the probabilistic hybrid.','tests/test_contextual.py':'Checks contextual features, model outputs, masks, gradients and checkpoint behaviour.','tests/test_contextual_selection.py':'Checks settings selected from external grouped predictions.','tests/test_contextual_consistency.py':'Checks agent/step agreement transformations.','__init__.py and src/__init__.py':'Mark Python package boundaries; contain no benchmark evidence.'}
table(['File','What it does'],list(modules.items()),[185,315])
page('22. File guide: reports, models and data')
table(['Folder / file family','Role'],[['data/Who&When/Algorithm-Generated/*.json','126 original generated-system conversations with reference labels.'],['data/Who&When/Hand-Crafted/*.json','58 original crafted-system conversations with reference labels.'],['data/stepfinder_safe_questions.json','Audited external question filter used for retained training cases.'],['artifacts/phase0...phase9/','Original phase reports, models, predictions, scope audit and verification.'],['benchmark_step/','Older ranker checkpoint, metrics and per-trace predictions; archived L2 setting preserved.'],['hybrid_upgrade/','Earlier hybrid model, grouped predictions, metrics and reports.'],['accuracy_study/','Grouped four-agent comparison output.'],['accuracy_push/','Preserved consistency, adaptation, subset and paper-scorer audits.'],['contextual_upgrade/','New checkpoints, selection, fold predictions, final predictions, metrics and reports.'],['results_graphs/','26 PNG figures, overview JPEG, CSV evidence, source hashes and graph guide.'],['docs/superpowers/specs/ and plans/','Architecture decisions and implementation plans.'],['output/pdf/','Updated review guide and PDF validation manifest.'],['README.md','How to run the project and what its results mean.'],['ALL_ACCURACY_RESULTS.md and BENCHMARK_STATUS.md','Consolidated accuracy history and paper comparison caveats.'],['RESEARCH_UPGRADE.md and ACCURACY_PUSH_PROMPT.md','Earlier experiments, remaining gaps and handoff for further research.'],['requirements*.txt and .gitignore','Dependency declarations and generated-cache exclusions.']],[240,260])
heading('Large files outside the project')
add('S:/pr/agentdiag-model-cache contains the frozen public encoder and packed vectors. S:/pr/StepFinder contains external source data. S:/pr/agentdiag-mlenv is the local ML environment. These are dependencies, not uploaded project-source files. Model weights inside contextual_upgrade are the small trained predictors.')
page('23. Live agents, multimodal claims and contribution')
heading('Do we need to run agents live?')
add('For the stated task of diagnosis from saved logs, training and testing on saved labelled records is valid. Live runs are needed if claiming live monitoring or demonstrating end-to-end integration with newly executed agents. Our project does not currently establish those claims.')
heading('Does this prove multimodal performance?')
add('No. The supplied labelled evaluation is based on text logs; raw image/audio/video diagnosis has not been tested. A project title or proposed feature does not replace evaluated multimodal inputs. A future extension needs modality-aware inputs and independent labels.')
heading('What can we call the contribution?')
add('A locally feasible, auditable failure-diagnosis implementation that combines task and conversation context, separate agent/step learning, probability blending, and traceable output evidence. The project also provides probabilistic missing-evidence reasoning and review support. This is a project contribution; the underlying methods are established, and worldwide novelty has not been proved.')
heading('What has not been established')
bullets(['A fully matched independent win over published papers.','Reliable early warning.','Live replay or counterfactual causal proof.','Tested image/audio/video diagnosis.','A Who&When Pro result.','A guaranteed faculty score of 18-19/20.'])
page('24. Review questions: purpose and methods')
qa=[('What is your project?','It investigates failed AI-team conversations and predicts the likely responsible agent and exact failure message.'),('How is this connected to probabilistic reasoning?','The Bayesian component combines evidence with prior probabilities, marginalizes missing observations, and checks confidence calibration.'),('Is the latest accuracy from the Bayesian network?','No. The 57.9% agent and 33.3% step scores are from the selected contextual ensemble. The Bayesian model remains a separate probabilistic component.'),('What is a trace?','The saved record of one task: messages, speaker names and task information.'),('What did you train?','Compact lexical and neural predictors. The pretrained Qwen embedding encoder was frozen.'),('Did you use cloud model APIs?','No for this local accuracy upgrade. Encoding and compact-model training ran locally.'),('Why separate agent and step prediction?','An agent can be identified from several messages, while locating an exact message is a different problem. Directly trained heads reduce dependence on a single message score.'),('Why use an ensemble?','Different predictors make different errors. Their selected probability mixture improved the generated-conversation result.'),('Why keep question groups together?','Related examples must not be scattered across training and validation, where they could make evaluation artificially easy.'),('Are labels hidden?','Prediction parses observed inputs without answer fields. Evaluation separately reads the correct labels to score predictions.')]
for q,a in qa: story.append(KeepTogether([p(q,'h2'),p(a)]))
page('25. Review questions: results and limitations')
qa=[('What improved?','Algorithm-Generated agents rose from 62/126 to 73/126; exact steps rose from 36/126 to 42/126.'),('Did every score improve?','No. Hand-Crafted agents stayed 37/58 and exact steps fell from 10/58 to 8/58. The older predictor is preferable for those steps.'),('Why is step accuracy lower?','There are many candidate messages, labels can mark a decisive earlier error, and exact scoring counts near misses as wrong.'),('Did you reach 60% and 35%?','No. We need at least 76/126 agents and 45/126 steps; current counts are 73 and 42.'),('Did you beat AgenTracer?','No. The cited Algorithm-Generated reference is 63.73% agent and 37.30% step, above our current values.'),('Are higher scores than Who&When a fair win?','They are numerical reference comparisons. A same-data, same-scorer independent evaluation is required for a defensible win claim.'),('How do you prove the numbers?','Provide per-trace predictions, count formulas, metric files, source hashes, saved model checkpoints and reproduction checks.'),('Is 76.9% your agent accuracy?','No. That is two-agent shortlist coverage for the Bayesian 13-case test, not the first-choice accuracy.'),('Is early warning reliable?','No. It gave 2/10 correct early warnings and wrong-agent warnings on 11/12 tested traces.'),('What would improve the project most?','More representative unseen labelled data, stronger long-context handling, calibrated predictions and a matched independent evaluation.')]
for q,a in qa: story.append(KeepTogether([p(q,'h2'),p(a)]))
page('26. Final presentation checklist')
bullets(['Describe the problem using a team-of-workers example.','State the actual input and output.','Explain saved-log testing and the two data groups.','Show the architecture and explain why context was added.','Give the exact old/new table, including the Hand-Crafted regression.','Separate Bayesian results from contextual benchmark results.','Show at least one successful prediction and one failure.','Use the charts and CSV evidence rather than unsupported claims.','State the limits and realistic next research steps.'])
heading('Short final statement')
add('"The implementation is reproducible and supports failure review. The contextual upgrade improved Algorithm-Generated agent and exact-step scores, but did not improve all subsets. The project provides traceable evidence, and further independent testing is needed before claiming published benchmark leadership."')
heading('Local versus GitHub')
add('Repository link: <link href="https://github.com/Sampath8989/pr-project/tree/main/agentdiag_v2" color="#246BFD">github.com/Sampath8989/pr-project/tree/main/agentdiag_v2</link>. Latest upgrades, graphs and this guide are saved locally. This PDF does not imply they have been pushed to main.')

for i in range(0,len(manifest['figures']),2):
    page(f'Appendix A. Result figures {i+1}-{min(i+2,len(manifest["figures"]))}')
    for item in manifest['figures'][i:i+2]:
        graph(item['file'],escape(item['file']+' | '+item['caption']))
    if i==0: add('These graphs summarize saved measurements. The full PNG files and CSV values are available under results_graphs/.')
page('Appendix B. Plain-English glossary')
table(['Term','Meaning'],[['Agent','One AI worker in a team.'],['Benchmark','A collection of labelled examples used to measure predictions.'],['Architecture','How the model components are arranged.'],['Embedding','A numeric representation of text meaning.'],['GRU','A small neural component that tracks a sequence.'],['Attention','A learned way to focus on relevant information.'],['Head','A model output section trained for a particular target.'],['Ensemble / blend','A weighted combination of several predictions.'],['Cross entropy','A training loss that rewards probability assigned to the correct answer.'],['Regularization','A restriction that discourages overfitting.'],['Epoch','One pass through the training examples.'],['Seed','A fixed starting value for reproducible random choices.'],['Group fold','A validation split that keeps related questions together.'],['Calibration','How well confidence matches actual correctness.'],['Brier / log loss / ECE','Probability-quality scores; lower values are better.'],['Marginalization','Sum over unknown possibilities rather than inventing an observation.'],['Confidence interval','An estimate of uncertainty around a measured score.'],['SHA-256','A file fingerprint used to detect changes.'],['Checkpoint','Saved trained model parameters.'],['Exploratory result','A useful development result that is not untouched independent confirmation.']],[145,355])
page('Appendix C. References and evidence sources')
refs=[('Who&When paper','https://arxiv.org/html/2505.00212'),('Who&When official benchmark repository','https://github.com/mingyin1/Agents_Failure_Attribution'),('AgenTracer paper','https://arxiv.org/html/2509.03312'),('StepFinder authors code','https://github.com/taiyu-zhu/StepFinder'),('Qwen3-Embedding-0.6B model card','https://huggingface.co/Qwen/Qwen3-Embedding-0.6B'),('Project GitHub','https://github.com/Sampath8989/pr-project/tree/main/agentdiag_v2')]
for title,url in refs: add(escape(title)+': <link href="'+url+'" color="#246BFD">'+escape(url)+'</link>')
add('Primary local sources: README.md; ALL_ACCURACY_RESULTS.md; BENCHMARK_STATUS.md; contextual_upgrade/Contextual_Report.md, Method_Comparison.md, metrics.json, selection.json, verification.json and Consistency_Report.md; original phase metrics; results_graphs source manifest; previous review guide. External sources were studied during the architecture upgrade. This PDF compiles those saved findings; it does not claim that all papers in the project library were freshly re-reviewed.')
heading('What this guide covers')
add('The original phase pipeline, historical and current predictors, local migration work, contextual accuracy changes, retained failed experiments, result evidence, source files, running commands and review preparation. It is not an exact transcript of every chat or an independent audit of unavailable experiments.')

# Inventory generated directly from disk, with specific source-file descriptions.
files=[x for x in ROOT.rglob('*') if x.is_file() and not any(part in {'.git','__pycache__','.matplotlib','output'} for part in x.relative_to(ROOT).parts)]
inventory=[]
for f in sorted(files):
    rel=f.relative_to(ROOT).as_posix()
    if rel.startswith('data/Who&When/') and f.suffix=='.json': role='Original labelled conversation; one trace used in evaluation.'
    elif rel in descriptions: role=descriptions[rel]
    elif rel in modules: role=modules[rel]
    elif f.suffix=='.py':
        try:
            doc=ast.get_docstring(ast.parse(f.read_text(encoding='utf-8')))
            role=(doc.splitlines()[0] if doc else 'Python implementation or experiment helper; inspect named functions for details.')
        except (SyntaxError,UnicodeError): role='Python source file.'
    elif f.suffix in {'.pt','.npz'}: role='Saved model parameters or numeric experiment data; used for reproduction.'
    elif f.suffix=='.jsonl': role='One saved record or prediction per line; supports evaluation and audit.'
    elif f.suffix=='.json': role='Structured metrics, configuration, model or audit output; interpreted by its folder.'
    elif f.suffix in {'.png','.jpg'}: role='Visual result chart or overview; see figure caption and source CSV.'
    elif f.suffix=='.csv': role='Tabular numerical or prediction evidence.'
    elif f.suffix=='.md': role='Human-readable documentation, results, planning or handoff record.'
    elif f.suffix in {'.pdf','.pptx'}: role='Review document or presentation; historical versions may predate this upgrade.'
    else: role='Project configuration, dependency declaration or supporting artifact.'
    inventory.append([rel,role])
page('Appendix D. Complete saved-file inventory','Generated from the current project folder. Repeated trace and prediction files share a role. Caches, Git internals and this output folder are excluded.')
table(['Relative path under agentdiag_v2','Purpose'],inventory,[260,240])

def chrome(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setFillColor(NAVY);canvas.rect(0,h-13,w,13,fill=1,stroke=0)
    canvas.setFillColor(TEAL);canvas.rect(0,h-13,100,13,fill=1,stroke=0)
    canvas.setFont('Review',8);canvas.setFillColor(colors.HexColor('#64748B'))
    canvas.drawString(48,25,'AGENTDIAG | UPDATED PROJECT REVIEW | OCTOBER 2026')
    canvas.drawRightString(w-48,25,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=43,title='AgentDiag Updated Project Review Guide',author='AgentDiag project',subject='Verified version comparisons and simple-English project guide')
doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
reader=PdfReader(OUT)
text='\n'.join(page.extract_text() or '' for page in reader.pages)
assert len(reader.pages)>25
assert '73/126 (57.9%)' in text and '42/126 (33.3%)' in text
assert '10/58 (17.2%)' in text and '8/58 (13.8%)' in text
assert 'swapp' not in text.lower()
assert len(rows)==184
for g in ['Algorithm-Generated','Hand-Crafted']:
    rr=[r for r in rows if r['subset']==g]
    assert sum(r['actual_agent']==r['predicted_agent'] for r in rr)==m['new'][g]['correct_agent']
    assert sum(r['actual_step']==r['predicted_step'] for r in rr)==m['new'][g]['correct_step']
checks={'created_at':datetime.now(timezone.utc).isoformat(),'pages':len(reader.pages),'figure_count':len(manifest['figures']),'inventory_files':len(inventory),'predictions_recounted':184,'accuracy_strings_verified':True,'pdf_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'source_metrics_sha256':hashlib.sha256((ROOT/'contextual_upgrade/metrics.json').read_bytes()).hexdigest(),'visual_review':'Pending rendering inspection'}
(DEST/'pdf_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(OUT),**checks},indent=2))
