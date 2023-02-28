# COMET-ATOMIC ja

We built a commonsense knowledge graph on events in Japanese, with reference to [ATOMIC](https://allenai.org/data/atomic) and [COMET](https://github.com/atcbosselut/comet-commonsense).
The graph was built from scratch, without translation.

We obtained the seed graph by Yahoo! Crowdsourcing and expanded it by in-context learning with HyperCLOVA JP.

## Data

The graph is in JSON Lines format.
Each line contains an event and its inferences for the four relation types, derived from those in ATOMIC.

We rearranged the relation types in ATOMIC by considering the two dimensions: *inference categories* and *time series*.
Therefore, the graph covers the following relations:

| \\     | Event   | Mental state |
| :----- | :------ | :----------- |
| Before | xNeed   | xIntent      |
| After  | xEffect | xReact       |

An example of the JSON objects is as follows:

```json
{
    "event": "Xが顔を洗う",
    "inference": {
        "event": {
            "before": [
                "Xが水道で水を出す"
            ],
            "after": [
                "Xがタオルを準備する",
                "Xが鏡に映った自分の顔に覚えのない傷を見つける",
                "Xが歯磨きをする"
            ]
        },
        "mental_state": {
            "before": [
                "スッキリしたい", 
                "眠いのでしゃきっとしたい"
            ],
            "after": [
                "さっぱりして眠気覚ましになる",
                "きれいになる",
                "さっぱりした"
            ]
        }
    }
}
```

## Models

We finetuned the Japanese [GPT-2](https://huggingface.co/nlp-waseda/gpt2-small-japanese) and [T5](https://huggingface.co/megagonlabs/t5-base-japanese-web) on the built graph.
The models are available at Huggingface Models:

- [nlp-waseda/comet-gpt2-small-japanese](https://huggingface.co/nlp-waseda/comet-gpt2-small-japanese)
- [nlp-waseda/comet-t5-base-japanese](https://huggingface.co/nlp-waseda/comet-t5-base-japanese)

For the GPT2-based model, special tokens for the four relation are added to the vocabulary.
Input a pair of a head and a special token to generate a tail.
Note that the head should be segmented into words by [Juman++](https://github.com/ku-nlp/jumanpp), due to the base model.

The T5-based model infers a tail with a prompt in natural language.
The prompts are different for each relation.

These two models were trained on 90% of the graph.
The evaluation results for the remaining 10% are as follows:

| Model         | BLUE  | BERTScore |
| :------------ | ----: | --------: |
| COMET-GPT2 ja | 43.61 | 87.56     |
| COMET-T5 ja   | 39.85 | 82.37     |

## Training

You can finetune models on the graph.
Note that the scripts are separated for GPT-2 and T5.

An example of finetuning GPT-2 is as follows:

```bash
pip install -r requirements.txt
python train_gpt2.py \
    --graph_jsonl graph_mrph.jsonl \
    --model_name_or_path nlp-waseda/gpt2-small-japanese \
    --output_dir comet_gpt2 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

## References

```bibtex
@InProceedings{ide_nlp2023_event,
    author =    "井手竜也 and 村田栄樹 and 堀尾海斗 and 河原大輔 and 山崎天 and 李聖哲 and 新里顕大 and 佐藤敏紀",
    title =     "人間と言語モデルに対するプロンプトを用いたゼロからのイベント常識知識グラフ構築",
    booktitle = "言語処理学会第29回年次大会",
    year =      "2023",
    url =       "https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B2-5.pdf"
    note =      "in Japanese"
}

@InProceedings{murata_nlp2023,
    author =    "村田栄樹 and 井手竜也 and 榮田亮真 and 河原大輔 and 山崎天 and 李聖哲 and 新里顕大 and 佐藤敏紀",
    title =     "大規模言語モデルによって構築された常識知識グラフの拡大と低コストフィルタリング",
    booktitle = "言語処理学会第29回年次大会",
    year =      "2023",
    url =       "https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B9-1.pdf"
    note =      "in Japanese"
}
```
