# Contextual attribution experiment

The configuration and blend were selected on external question-group validation. The final training used 2,004 filtered external traces from 58 questions; the original Who&When labels were not training targets.

| Method | Algorithm agent | Algorithm exact step | Hand-crafted agent | Hand-crafted exact step |
| --- | ---: | ---: | ---: | ---: |
| Previous local ranker | 62/126 (49.2%) | 36/126 (28.6%) | 37/58 (63.8%) | 10/58 (17.2%) |
| Selected contextual candidate | 73/126 (57.9%) | 42/126 (33.3%) | 37/58 (63.8%) | 8/58 (13.8%) |
| Who&When GPT-4o all-at-once, no answer supplied (paper) | 51.12% | 13.53% | 53.44% | 3.51% |
| AgenTracer, no answer supplied (paper) | 63.73% | 37.30% | 63.82% | 20.68% |

Paper sources: [Who&When](https://arxiv.org/html/2505.00212), [AgenTracer](https://arxiv.org/html/2509.03312). These are reported paper scores, not a matched head-to-head comparison.

## Frozen selection

```json
{
  "selection": {
    "agent": {
      "candidate": "mix__temporal_exact_16__sparse_0.5",
      "weight": 0.5,
      "selection_score": [
        0.7896865520728008,
        0.7586206896551724,
        -0.5
      ]
    },
    "step": {
      "candidate": "sparse_0.5",
      "weight": 0.75,
      "selection_score": [
        0.6885743174924166,
        0.6472906403940887,
        -0.75
      ]
    }
  },
  "candidate_names": [
    "baseline",
    "sparse_0.5",
    "sparse_2.0",
    "temporal_exact_8",
    "temporal_exact_16",
    "temporal_exact_24",
    "temporal_distance_8",
    "temporal_distance_16",
    "temporal_distance_24",
    "attention_exact_8",
    "attention_exact_16",
    "attention_exact_24",
    "mix__temporal_exact_8__sparse_0.5",
    "mix__temporal_exact_8__sparse_2.0",
    "mix__temporal_exact_16__sparse_0.5",
    "mix__temporal_exact_16__sparse_2.0",
    "mix__temporal_exact_24__sparse_0.5",
    "mix__temporal_exact_24__sparse_2.0",
    "mix__temporal_distance_8__sparse_0.5",
    "mix__temporal_distance_8__sparse_2.0",
    "mix__temporal_distance_16__sparse_0.5",
    "mix__temporal_distance_16__sparse_2.0",
    "mix__temporal_distance_24__sparse_0.5",
    "mix__temporal_distance_24__sparse_2.0",
    "mix__attention_exact_8__sparse_0.5",
    "mix__attention_exact_8__sparse_2.0",
    "mix__attention_exact_16__sparse_0.5",
    "mix__attention_exact_16__sparse_2.0",
    "mix__attention_exact_24__sparse_0.5",
    "mix__attention_exact_24__sparse_2.0"
  ],
  "selected_external_cv_metrics": {
    "Algorithm-Generated": {
      "traces": 989,
      "correct_agent": 781,
      "agent_accuracy": 0.7896865520728008,
      "correct_step": 681,
      "step_accuracy": 0.6885743174924166,
      "correct_both": 608,
      "joint_accuracy": 0.6147623862487361
    },
    "Hand-Crafted": {
      "traces": 1015,
      "correct_agent": 770,
      "agent_accuracy": 0.7586206896551724,
      "correct_step": 657,
      "step_accuracy": 0.6472906403940887,
      "correct_both": 572,
      "joint_accuracy": 0.5635467980295567
    },
    "Overall": {
      "traces": 2004,
      "correct_agent": 1551,
      "agent_accuracy": 0.7739520958083832,
      "correct_step": 1338,
      "step_accuracy": 0.6676646706586826,
      "correct_both": 1180,
      "joint_accuracy": 0.5888223552894212
    }
  },
  "external_source": {
    "source": "https://github.com/taiyu-zhu/StepFinder",
    "commit": "48d71c7090adca3667a31b043a78484f407b4a2f",
    "question_filter": "Token Jaccard below 0.85, then BGE-small-en-v1.5 maximum cosine to every original Who&When question below 0.75",
    "safe_question_sha256": [
      "009a82f5a7f10732a960854f63adb14ae32cb362e31e875efd2b143fbb4bf078",
      "012af48eddfacdc154aeaa6c562cdcc7f016144e4e9de6bf1c63312d329d1803",
      "0e3f3d6821f4ba7c095a612d4254bd469d3fced66bdc887606809f9f4e1559cd",
      "0f7acf0aaea4df48742a07e31d1489a9bfb31fa0bb5c7ea87dbbc91c1dd0f393",
      "1141f27e5857da4de39302a9749fe2715649727602b931f153fb12e8ca4f7f1f",
      "159d62bbd588bd955b49d712dcd648345283985576b7e65906e217f6a058a923",
      "1ab97e01e679c7d48d267f7be7fb9822cf58cce8d730b7f3bd5cad9786500257",
      "22618c1758630f4b3614606ac7edf0bb1393cb3f6046e0e7824f20b8ce4b580f",
      "233bd2df14bb6a3f2bd3d7d791d0ccde2521a21b6320a2bba270da2fcf0fc0fe",
      "319894ef2e18db1c8205a6eaf474398d295548a7736adfdc05f7c8d59d088217",
      "339683ae59840571f897b3a151e61114f741c5921b942e169809398aac5a20b3",
      "36d3b0ed1232e2b8549801baaddb5be7c7f09bf16c859c3b5849767c40099ad6",
      "3898a0d98e7aaa5ff227d1d52a31402df8b6f84c4eff962d172a6fdd467129fb",
      "3b3f262380ad77b125b3c1b0d1b07b870ac1800674c3e3dbfdc01d433ad1ac61",
      "3b58acc413a67c31bc564440312149fc2dd99e92d226d4ec5bb71965c9db1f33",
      "4588ce209ce7922e1ce4d82770d3a1ed89a9ef3b66ec15386dce4905c5cf8827",
      "48e000664ff613fd4e78d0ca7044665046fd78ddc2eaea2bf6111d72fe61e823",
      "4cd0919f95307894cd8719806fdb0818b0dcb27c22ff8bfbb035ba98fa310d28",
      "56a7a534c6735d76df67198405b9a8bd0daa581fc03220ebc5d3a85047aaabe6",
      "5a146a7abfad6c96ee495e2168011ed4d216ce9a0cf24f90d8011a0f5ece0a34",
      "63d90c00efe12208b16ba7d85d14af348db8c2f2b3b52cd4319390c4444c378b",
      "6c4e131c0de616e64d667f3f1469bc340495b09ad193fd9aef0fe2e2d2c02b15",
      "6f77780b4a9ded618d37cccb47e94b4989e8cc81cded1cdbebb42e7052b6d069",
      "708eeb2dcd4d00d0c801fc229fbdff7b68f7e1ec853de89f6db04b5d598c1085",
      "70bb82602d052a12e91860f5f99086960705e7bde39ae3fd54939492616866de",
      "729384c416bb9a7a49f4e21a30d06fbedac3da8f9f3644ad16405d20b5a099b4",
      "7a2d441e7b205a98c5c12708326ee6b921cc5a628d762b372254d45ac87a80c9",
      "7e697827996867db4920779518e9c1f56d3867f208437771daae069a7f1303d0",
      "7fa0c5cfbf4c6bfc84b1d021000f7d0fb5816825ffed1d11573ae33dbb187f07",
      "7fab70f972a3907b64e19181958ade4496a6fce7ad6931b43499128395d2a925",
      "82e75b3516de083a2889a4cb7a4bed73c6a4f948616247ebfa9e90b4be8bc5ff",
      "8d919fccfb41765d0611175a06e2600ca994c1114f0c3bb1f3e2a35dcfca1771",
      "8e96f13e974a7c302f38000d6764a58129df08adfd2c21fef2a6f5ea2af4d2e2",
      "8fa42360185068216f2919935148d4e1ad28ddc18da0abd0f4bb0b6b6f84b127",
      "90f300f8203fb8b74c3ac0a3abf72b747ec0b10dab1a4d0b95446a0118497c07",
      "92f24644aee38612c92bab077b96ec3e0b3d6cccb07c2ff5442e72378a764e7b",
      "957a70e9b6a95f9fb5871633b08bba5e8463af333ce5eba7d09283eed189cb38",
      "95b4a6c4499061d3a2618efb1ed0af16211f6fa60bf86e0253f356c340a93e18",
      "999ee9faee889df8bbdedb4029fe81ea7c1cd2bde554601da21bb3a49fa17047",
      "9a4509f4e50824e48642a947ca36657062b8344d14a78d7ba352c2f031ab0a7c",
      "a3d56cbb2336e96126db201d544e23e6ec3092fc6608b2bf2eba14da672388e1",
      "a734e6022ed411c5170dc7d8dcb1da34c3977194501243d43d21c9192028aa0b",
      "abe10d28fe39bf6cf62411f0c12e1eaff24bbb94194621e1b6f244c28a40df60",
      "af673b7b4ff149f0afda498a17b2eca5244a4730cbd0d7d7111324a6390eef42",
      "b0260b8d170d9b81fc7c5023b5299b0034825de85f1764384b69a02e24985c47",
      "b2d27c8cd36dea762c857c2596b940d43930948cd68a4cb9a9922f909c7fd84c",
      "b4665ea53eee7bbab1139cd9c62ca39c309fa5f29061c4c0853fac3ee28fb736",
      "c10608e12a9d22f47fd6bc5c069ac60c902d1832bfc16fc88336c7a2f3337d87",
      "c3cda75e7f8d97e7d6ea89acd7daad352257c75254921fa8528871f12db4b163",
      "c407cd65a761a2f37306a17aad64ffaa2a7c1f75a702c291d1717f7d7d088cf7",
      "c65b3f4b3bf15ad490a2ed70438560e0f4a7aea89fdd1d9b0289ea3778058ea8",
      "deca4b10e362b100f212c65916fb2fda1b5c3c0d8de1b5bb4aa919df55d23cd4",
      "df0ddda51c98dd6501f904d50f09532b7b74327c6394e60d165ffc1277a948b9",
      "e1979ab3ea4b3fbd1f71b54d78778b0f292a4903c4da6fdecdebbc7540e18153",
      "e3c34e7b0e245f654330995991e35c064b347095e69e12ddcd8eca7096654873",
      "ecd99b2a5bbbc50f0ee8bf46be11715030abdf864cf9fc4aa82a3d1621745c61",
      "fb9ba3ab6a13d0adc677f993e90d54914a5cdf211305a1bba6bf16ec4ccb9b7c",
      "fd7afe8669662343df9ffb693bab6dc24fbebf052e32e86bca986d6e7967866b"
    ]
  },
  "seeds": [
    13,
    29
  ],
  "epochs": [
    8,
    16,
    24
  ],
  "configs": {
    "temporal_exact": {
      "model": {
        "hidden": 64,
        "dropout": 0.25,
        "temporal": true
      },
      "sigma": 0
    },
    "temporal_distance": {
      "model": {
        "hidden": 64,
        "dropout": 0.25,
        "temporal": true
      },
      "sigma": 1
    },
    "attention_exact": {
      "model": {
        "hidden": 64,
        "dropout": 0.25,
        "temporal": false
      },
      "sigma": 0
    }
  },
  "training_trace_ids_sha256": "87d4b9fb99d7e0354079b41ea86cca24ff58681a039ffadb00941e5afe289f4b",
  "protocol": "Head-specific blending selected only on external question-group OOF accuracy. Algorithm-Generated primary, Hand-Crafted tie breaker. No Who&When target labels selected this configuration.",
  "cv_file_sha256": {
    "cv_attention_exact_13_16.jsonl": "7caadf150cb140526c036934757550d7951461f7b16b4532cd3c049ec51b4d21",
    "cv_attention_exact_13_24.jsonl": "383efff2f75eabbdea4cb1479c4e5a379d264c187dac364033d358cf3acbc063",
    "cv_attention_exact_13_8.jsonl": "c7aff67422661b32e4ed312a0ed7324662c2e7cdc84ba912c80d0f6d5ff07dc6",
    "cv_attention_exact_29_16.jsonl": "576a6081dbeffa87108f70b58023a3df81bfe686d461752f38b0269fa0f13106",
    "cv_attention_exact_29_24.jsonl": "c16974e2a5575c6c8517ccfa78edc472011652c49b728ca5a660f432786fecf7",
    "cv_attention_exact_29_8.jsonl": "d24709c3e9e238f123d299527a82ee029a5f900a359e05f451aea31b562d2e70",
    "cv_selected.jsonl": "0166653e3dd37b2527423b4a977cd499566b9c2c7db7c63cfe51992fa62068c2",
    "cv_sparse_0.5.jsonl": "a5d615eabb609e916e5070cee38b6407e150a9fdda3daac0ae718d54f1380bab",
    "cv_sparse_2.0.jsonl": "09f2ba725ea7ea1da65ee8f0e1e96bab77b43a1dfc9edd4978019a9cb72c8634",
    "cv_temporal_distance_13_16.jsonl": "cd4f34a6a8f41f0947b95fc9db66b2a9786211c8d20c50137698495d0abb3a95",
    "cv_temporal_distance_13_24.jsonl": "610477005351b68d36af229168cf58e4081030a6d753092e54aade275808c1f7",
    "cv_temporal_distance_13_8.jsonl": "c03f1179e045892510a0556c269dc9b15b20527a49dca3c3175c27d7046c73c4",
    "cv_temporal_distance_29_16.jsonl": "10e04b2e142fdd61810760bce8ed2f2672d8039038c52d3f7acd7796ac85fc4e",
    "cv_temporal_distance_29_24.jsonl": "5a8fed60f6c07f0493a04c5571b0b59dd6e290a8bdec68b753d27fac88463262",
    "cv_temporal_distance_29_8.jsonl": "1a8338a33c1bac28a7b8f777ae32a324dd6938ec9fd840f096897903877a553c",
    "cv_temporal_exact_13_16.jsonl": "bb37a5d53dfd267ac106334696cfbb84a3e52b02242012b483a068e7519b0cc0",
    "cv_temporal_exact_13_24.jsonl": "e4bb8be76a1b962285a949e5292c12e9ebee64432de530f914707ac1e529f9f3",
    "cv_temporal_exact_13_8.jsonl": "46251a528395d6af3b4d5b93e0e5e0a6d9805814fd903c4ce5ffc797404fbff7",
    "cv_temporal_exact_29_16.jsonl": "9b4b124e62eb7e7797085e9760c60ef39147386c69bfd43a0040b93a8187dd2d",
    "cv_temporal_exact_29_24.jsonl": "3249cb81be98f98aaed8cb4e3ff8548d65e2c65737148c40fe718e75efbc36aa",
    "cv_temporal_exact_29_8.jsonl": "89449513917ff9f1383e3f2c6ee95660aea3da54eedd4232937c79364316f929"
  },
  "baseline_cv_sha256": "40afb4fe667d8a8bdc945bf012d0ef7835f048d5a3b335a6450da5360bfc9ce7",
  "fold_manifest_sha256": "ecce35a4ca5e47ad287f75bc174bf4173de099925d1beeae90340ddac685d938",
  "source_code_sha256": {
    "src/data.py": "9e8648f86d124579a2c328220714dc4c427bfc5262864b652d3178bc23dbc10b",
    "src/contextual_ranker.py": "76bddefedeaaf3b2b54bc9fa5eb34f26ab7fbdfc34ae09bf70d47de0063ec8b7",
    "src/contextual_sparse.py": "02e67280e257b50b17be15d5e303b0f1f116fd76ea0199ff9fbdb486d8684cdc",
    "src/semantic_encoder.py": "c6ac56bff754cc780e8e0d2bd44e8bd89e6c3bdd5206db8dc7d08a7b5d125f0b"
  }
}

```

## Paired changes

```json
{
  "Algorithm-Generated": {
    "agent": {
      "gains": 16,
      "losses": 5,
      "two_sided_exact_p": 0.02660369873046875
    },
    "step": {
      "gains": 11,
      "losses": 5,
      "two_sided_exact_p": 0.210113525390625
    }
  },
  "Hand-Crafted": {
    "agent": {
      "gains": 3,
      "losses": 3,
      "two_sided_exact_p": 1.0
    },
    "step": {
      "gains": 1,
      "losses": 3,
      "two_sided_exact_p": 0.625
    }
  }
}
```

## What changed

The lexical comparator includes word pairs, preceding and subsequent messages, and an agent classifier trained directly on agent labels. The semantic candidates use a pinned Qwen3-Embedding-0.6B encoder, question interactions, contextual attention, optional bidirectional GRU, and separately supervised agent/step heads. Selection may retain only some components; the selection manifest specifies the model actually used.

## Verification and limits

All 184 checkpoint predictions were reproduced from label-free parsing. Exact metric counts and probability validity were checked. The model analyzes saved text logs; it does not run live agents, counterfactual replay or multimodal inputs. The original benchmark has been inspected during development. These exploratory results cannot establish independent benchmark leadership. Matching reported percentages does not establish statistical equivalence or identical scoring. The Bayesian network remains a separate probabilistic reasoning component, not the model responsible for these scores.
