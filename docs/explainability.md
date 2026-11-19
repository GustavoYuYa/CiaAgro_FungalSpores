# Explicabilidade: EigenCAM e EigenGradCAM

Os novos scripts reutilizam as arquiteturas, checkpoints e transformações existentes.
Nenhum treinamento ou arquivo anterior precisa ser alterado. Há dois pontos de entrada:

```powershell
python -m scripts.explain_eigencam --dataset-root "C:\GitHub_CiaAgro\dataset_huggingface"
python -m scripts.explain_eigengradcam --dataset-root "C:\GitHub_CiaAgro\dataset_huggingface"
```

Execute a partir da raiz do repositório, no ambiente de pesquisa com `torch`,
`torchvision`, `einops`, `opencv-python` (ou OpenCV do Conda), `numpy`, `pandas`,
`Pillow`, `grad-cam` e `openpyxl`. As dependências já são contempladas pelo ambiente
descrito no projeto. O pacote é instalado como `grad-cam` e importado como
`pytorch_grad_cam`. Não há necessidade de Node.js ou serviços externos.

Por padrão cada comando avalia M1–M4 de CNN e SimpleViT, nos dois cenários,
usando todos os índices do teste. Não há filtro por acertos.

Para uma execução exploratória pequena:

```powershell
python -m scripts.explain_eigengradcam --dataset-root "D:\meu_dataset" --architectures cnn simplevit --experiments M1 --max-samples 5 --deletion-steps 20 --device cpu
```

`--max-samples` seleciona os primeiros N índices, não uma amostra estratificada.
Remova esse argumento para os resultados finais. O índice original é preservado,
inclusive para gerar o ruído. Deletion AUC demanda várias inferências por imagem:
o padrão usa 100 intervalos (101 avaliações, incluindo os extremos), agrupadas em
lotes de até 32. Reduza `--deletion-batch-size` se necessário para a memória da GPU.

## Entradas

- Checkpoints: `experiments/{cnn,simplevit}/{M1,M2,M3,M4}/checkpoint_best.pt`.
- Splits: `DATASET_ROOT/splits/{normal,segmented_black}/test_{files,labels}.npy`.
- Imagens originais e segmentadas sobre fundo preto referenciadas nos splits.

Use `--experiments-root`, `--splits-root` e `--output-root` para outros caminhos.
O programa verifica comprimentos, rótulos, nomes e ordem dos pares e existência
das imagens. Os rótulos/nome de classe vêm dos splits e pastas; o mapeamento antigo
de quatro classes dos notebooks não é reutilizado. A normalização é lida de cada
checkpoint, sem recalcular estatísticas no teste. Imagens permanecem em 64×64.

## Métodos e classe-alvo

- **EigenCAM:** projeção principal das ativações; não depende da classe-alvo.
- **EigenGradCAM:** projeção principal de ativações multiplicadas pelos gradientes
  do logit da classe-alvo.
- CNN: última convolução, `block4[3]`, antes de BatchNorm/ReLU e do último pooling,
  com resolução espacial nativa 8×8.
- SimpleViT: `transformer.layers[-1][0].to_out`, como no notebook de referência.
  Os tokens são reorganizados em uma grade espacial (16×16 nos checkpoints atuais).
  Este SimpleViT não possui CLS token, portanto nenhum token é removido.

O padrão `--target true` mantém a definição quantitativa de `scriptXAI.ipynb`:
nos erros, as métricas também se referem à classe real. `--target predicted`
permite investigar a decisão efetivamente tomada. Em ambos os casos a classe-alvo
do Deletion fica fixa durante toda a curva; ela não acompanha mudanças no argmax.
Para EigenCAM, a opção afeta a classe acompanhada no Deletion, mas não o mapa.
Cada execução registra a política utilizada e salva o mapa efetivamente medido.

Não são usados augmentation smoothing ou um segundo processamento PCA.
Nas implementações Eigen do pacote, a própria geração já usa a projeção principal.

## Máscara e cenários

`PairedTestDataset` reutiliza `EsporosDataset`, `EsporosDatasetNoise` e
`transform_wout_normalize()`. A máscara segue exatamente a expressão do dataset:

```python
mask = (segmented.max(dim=0, keepdims=True).values > 0).float()
```

`segmented` é a imagem sobre fundo preto, transformada em tensor **depois** do
Resize de torchvision, **antes** do ruído e da normalização. Não se calcula uma
nova segmentação; não se redimensiona uma máscara binária criada antes do Resize.
Pixels de borda ou resíduos de compressão que atendam a `> 0` são incluídos,
exatamente como na implementação atual. A mesma máscara pareada serve a Clean
e Noise; a máscara não limita os pixels removidos no Deletion.

Noise é obtido pelo dataset existente: `RandomState(indice_original)`, ruído
gaussiano com média 0,5/desvio 0,2, limitado a [0,1], em formato CHW. Não são
utilizadas as outras distribuições presentes no notebook antigo.

## Métricas

1. **Energia interna:** `sum(CAM * mask) / sum(CAM)`, como no notebook.
   Quanto maior, maior a fração de ativação dentro da máscara. É registrada
   também a fração da imagem ocupada pela máscara, útil para interpretar o valor.
2. **IoU:** interseção/união entre a máscara e `CAM >= percentile(CAM, 90)`.
   `--iou-percentile` ajusta o percentil. O operador `>=` mantém os empates do
   notebook: a região selecionada pode exceder 10% da imagem. A fração efetivamente
   selecionada e o limiar ficam nas métricas por imagem.
3. **Deletion AUC:** ordena pixels espaciais por CAM decrescente e substitui os
   três canais de cada pixel. Integra, pelo trapézio, a probabilidade softmax da
   classe fixa em função da fração real removida, de 0 a 1. Inclui imagem intacta
   e totalmente substituída. Não divide a curva pela probabilidade inicial.
   Empates usam ordem estável de índice do pixel. Menor AUC indica queda mais
   rápida de confiança **para o protocolo escolhido**.

O preenchimento padrão é `--deletion-baseline mean`: média RGB de treinamento do
checkpoint, aplicada antes da normalização (equivale a zero normalizado).
Alternativas: `black` (RGB zero) e `gaussian` (0,5/0,2, limitado a [0,1], fixo
por índice e reutilizado ao longo da curva). As médias variam entre M1–M4;
para comparar com o mesmo preenchimento RGB entre modelos, use explicitamente
`black` ou `gaussian`. Não misture protocolos diferentes numa mesma comparação.

CAMs totalmente nulos recebem energia e IoU iguais a zero, evitando o IoU
artificial de uma imagem inteira selecionada. Máscaras vazias recebem métricas
de localização ausentes (`NaN` no CSV, célula vazia no Excel). Essas duas exceções
ao caso usual dos notebooks ficam sinalizadas por `cam_zero` e `mask_valid`.
O Deletion ainda é calculado; CAMs constantes são sinalizados (`cam_constant`),
pois sua ordem de remoção decorre somente do desempate.

## Saídas separadas

Cada script cria uma pasta de execução única. `--run-name nome` permite nomeá-la;
um nome já existente é rejeitado para evitar misturar ou sobrescrever execuções.

```text
experiments/xai/
  EigenCAM/<execucao>/
    cnn/cnn_EigenCAM_Metrics.xlsx
    simplevit/simplevit_EigenCAM_Metrics.xlsx
  EigenGradCAM/<execucao>/
    cnn/cnn_EigenGradCAM_Metrics.xlsx
    simplevit/simplevit_EigenGradCAM_Metrics.xlsx
```

São **quatro arquivos Excel** ao executar os dois métodos nas duas arquiteturas.
Cada arquivo reúne M1–M4 sem agregar os modelos entre si:

- `Summary`: média, desvio amostral, contagens válidas/ausentes por modelo,
  cenário e métrica.
- `By_Class` e `By_Correctness`: mesmos resumos por classe real e acerto/erro.
- `Clean` e `Noise`: valores por imagem, classes, confiança, diagnóstico do CAM
  e caminho da visualização.
- `Protocol`: parâmetros, normalização, camada, versões, seleção e caminhos.

Dentro de cada arquitetura há `{M1,...,M4}/{Clean,Noise}/`, com:

- `heatmaps/*.png`: imagem original pareada à esquerda e CAM sobre essa mesma
  original à direita. **No cenário Noise, a inferência usa ruído, mas a imagem
  de referência exibida é Clean**, conforme a solicitação. Título identifica
  cenário, método, classe-alvo, classe real e predição.
- `cams/*.npy`: mapas numéricos em 64×64, sem quantização ou colormap.
- `metrics.csv`: resultados por imagem, gravados progressivamente.
- `deletion_curves.csv`: pontos das curvas para auditoria do AUC.

`protocol.json` registra o estado `running` ou `complete` por modelo. O Excel é
atualizado ao concluir cada modelo. Uma execução interrompida pode ter CSVs
parciais; não interprete uma pasta incompleta como resultado de M1–M4 completos.

## Organização e verificação

`src/explainability/` separa dados, modelos/camadas, geração CAM, métricas,
relatórios e orquestração. Os dois scripts em `scripts/` são pontos de entrada.

```powershell
python -m unittest discover -s tests -p test_explainability.py -v
```

Os testes verificam equivalência exata com os datasets existentes, pareamento,
fórmulas de localização, ordem/normalização/alvo fixo no Deletion, reshape sem
perda de tokens, funcionamento dos dois CAMs e liberação de hooks.

As máscaras são referências automáticas da segmentação, não anotações biológicas
independentes. Sobreposição espacial não comprova, sozinha, fidelidade da
explicação. AUC pode ser baixo porque a classe real já tinha baixa confiança:
consulte as probabilidades iniciais e os resumos por acerto/erro. Mapas de baixa
resolução são ampliados pelo pacote antes das métricas em 64×64. Resultados
destes dois cenários não demonstram invariância a todo tipo de fundo.

Referências:
- [Documentação do pytorch-grad-cam](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)
- [Adaptação para Transformers](https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html)
- [Discussão de métricas de explicação](https://jacobgil.github.io/pytorch-gradcam-book/CAM%20Metrics%20And%20Tuning%20Tutorial.html)
- [RISE: métricas deletion/insertion](https://arxiv.org/abs/1806.07421)
- Notebooks locais `arquivos-externos/scriptXAI.ipynb` e `scriptMetricasXAI.ipynb`.
