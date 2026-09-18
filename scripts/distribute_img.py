from pathlib import Path
import shutil

PASTA_ORIGEM = Path(r"C:\GitHub_CiaAgro\CiaAgro_FungalSpores\data\dataset_huggingface\cropped_images\phakops")
PASTA_DESTINO = Path(r"C:\GitHub_CiaAgro\CiaAgro_FungalSpores\data\dataset_huggingface\cropped_images\Phakopsora_pachyrhizi")

MAX_IMAGENS_POR_PASTA = 5000

# False = COPIA as imagens (mais seguro)
# True  = MOVE as imagens
MOVER_ARQUIVOS = False

# Extensões consideradas como imagens
EXTENSOES = {
    ".jpg",
    ".jpeg",
    ".png"
}

def organizar_imagens(pasta_origem, pasta_destino, max_por_pasta=1000, mover=False):

    if not pasta_origem.exists():
        raise FileNotFoundError(
            f"A pasta não existe:\n{pasta_origem}"
        )

    imagens = sorted(
        [
            arquivo
            for arquivo in pasta_origem.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in EXTENSOES
        ]
    )

    total = len(imagens)

    print("=" * 60)
    print("ORGANIZAÇÃO DO DATASET")
    print("=" * 60)
    print(f"Pasta: {pasta_origem}")
    print(f"Imagens encontradas: {total}")
    print(f"Máximo por pasta: {max_por_pasta}")
    print(f"Modo: {'MOVER' if mover else 'COPIAR'}")
    print("=" * 60)

    if total == 0:
        print("Nenhuma imagem encontrada.")
        return

    numero_pastas = (total + max_por_pasta - 1) // max_por_pasta

    print(f"Subpastas necessárias: {numero_pastas}")
    print()

    for i in range(numero_pastas):

        inicio = i * max_por_pasta
        fim = min(inicio + max_por_pasta, total)

        imagens_lote = imagens[inicio:fim]

        # Exemplo: 000, 001, 002...
        nome_subpasta = f"{i:03d}"
        subpasta = pasta_destino / nome_subpasta

        subpasta.mkdir(exist_ok=True)

        print(
            f"[{i + 1}/{numero_pastas}] "
            f"{nome_subpasta}: {len(imagens_lote)} imagens"
        )

        for imagem in imagens_lote:

            destino = subpasta / imagem.name

            # Evita sobrescrever arquivos existentes
            if destino.exists():
                print(f"  Já existe: {imagem.name}")
                continue

            if mover:
                shutil.move(str(imagem), str(destino))
            else:
                shutil.copy2(str(imagem), str(destino))

    print()
    print("=" * 60)
    print("CONCLUÍDO")
    print("=" * 60)

organizar_imagens(
    PASTA_ORIGEM,
    PASTA_DESTINO,
    max_por_pasta=MAX_IMAGENS_POR_PASTA,
    mover=MOVER_ARQUIVOS
)