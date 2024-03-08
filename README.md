# COMET-ATOMIC ja

We built a commonsense knowledge graph on events in Japanese, with reference to [ATOMIC](https://allenai.org/data/atomic) and [COMET](https://github.com/atcbosselut/comet-commonsense).
The graph was built from scratch, without translation.

We obtained the seed graph by Yahoo! Crowdsourcing and expanded it by in-context learning with HyperCLOVA JP.

## Data

The graphs are in JSON Lines format.
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

`graph.jsonl` is the original graph built in [this paper](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B2-5.pdf), while `graph_v2.jsonl` is the larger one expanded in [this paper](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/B9-1.pdf).
The graphs with `mrph` in their filename have the triples whose head and tail were segmented into words by [Juman++](https://github.com/ku-nlp/jumanpp).

The original graph and `v2` have 1,471 and 1,429 unique heads, respectively.
The numbers of unique triples in their graphs are as follows:

| Relation | Original |     V2 |
| :------- | -------: | -----: |
| xNeed    |    9,403 | 44,269 |
| xEffect  |    8,792 | 36,920 |
| xIntent  |   10,155 | 52,745 |
| xReact   |   10,941 | 60,616 |

For the original graph, ten inferences were generated for each event and relation.
`v2` was expanded by generating ten times as many inferences, i.e., 100 inferences for each event and relation.
Note that in both graphs, duplicated triples were removed.

## Models

We finetuned the Japanese [GPT-2](https://huggingface.co/nlp-waseda/gpt2-small-japanese) and [T5](https://huggingface.co/megagonlabs/t5-base-japanese-web) on the built graph.
The models are available at Huggingface Models:

- [`nlp-waseda/comet-gpt2-small-japanese`](https://huggingface.co/nlp-waseda/comet-gpt2-small-japanese)
- [`nlp-waseda/comet-v2-gpt2-small-japanese`](https://huggingface.co/nlp-waseda/comet-v2-gpt2-small-japanese)
- [`nlp-waseda/comet-t5-base-japanese`](https://huggingface.co/nlp-waseda/comet-t5-base-japanese)

Note that `v2` models were finetuned on the expanded graph.

For the GPT2-based model, special tokens for the four relation are added to the vocabulary.
Input a pair of a head and a special token to generate a tail.
Note that the head should be segmented into words by Juman++, due to the base model.

The T5-based model infers a tail with a prompt in natural language.
The prompts are different for each relation.

These two models were trained on 90% of the graph.
The evaluation results for the remaining 10% are as follows:

| Model            |  BLUE | BERTScore |
| :--------------- | ----: | --------: |
| COMET-GPT2 ja    | 43.61 |     87.56 |
| COMET-GPT2 ja v2 |       |           |
| COMET-T5 ja      | 39.85 |     82.37 |

COMET-GPT2 ja v2 will be evaluated soon.

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
@misc{ide2023phalm,
      title={PHALM: Building a Knowledge Graph from Scratch by Prompting Humans and a Language Model}, 
      author={Tatsuya Ide and Eiki Murata and Daisuke Kawahara and Takato Yamazaki and Shengzhe Li and Kenta Shinzato and Toshinori Sato},
      year={2023},
      eprint={2310.07170},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
