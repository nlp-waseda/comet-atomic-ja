# COMET-ATOMIC ja

We build a commonsense knowledge graph on events in Japanese, with reference to [ATOMIC](https://allenai.org/data/atomic) and [COMET](https://github.com/atcbosselut/comet-commonsense).
The graph is built from scratch, without translation.

## Data

The graph is in JSON Lines format.
Each line has an event and its inferences for the four relation types, derived from those in ATOMIC.

| \\ | Event | Mental state |
| - | - | - |
| Before | xNeed | xIntent |
| After | xEffect | xReact |

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

We finetune the Japanese [GPT-2](https://huggingface.co/nlp-waseda/gpt2-small-japanese) and [T5](https://huggingface.co/megagonlabs/t5-base-japanese-web) on the built graph.

- [nlp-waseda/comet-gpt2-small-japanese](https://huggingface.co/nlp-waseda/comet-gpt2-small-japanese)
- [nlp-waseda/comet-t5-base-japanese](https://huggingface.co/nlp-waseda/comet-t5-base-japanese)

For the GPT-2-based model, special tokens for the four relation are added to the vocabulary.
Input a pair of a head and a special token to generate a tail.

The T5-based model infers a tail with a prompt in natural language.
The prompts are different for each relation.

## Reference

```bib
@InProceedings{ide_nlp2023_event,
    author =    "井手竜也 and 村田栄樹 and 堀尾海斗 and 河原大輔 and 山崎天 and 李聖哲 and 新里顕大 and 佐藤敏紀",
    title =     "人間と言語モデルに対するプロンプトを用いたゼロからのイベント常識知識グラフ構築",
    booktitle = "言語処理学会第29回年次大会",
    year =      "2023",
    url =       ""
}
```
