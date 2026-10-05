from pathlib import Path


def corrigir_nomes_classes():
    # Caminhos relativos seguindo sua estrutura (data/scripts/)
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent
    whole_slide_dir = data_dir / "whole_slide"
    
    # O que procurar e pelo que substituir
    nome_errado = "Fusarium graminearumm"
    nome_correto = "Fusarium graminearum"
    
    arquivos_corrigidos = 0
    linhas_corrigidas = 0
    
    print(f"Buscando por '{nome_errado}' em {whole_slide_dir}...\n")
    
    # Varre todos os arquivos .txt
    for txt_file in whole_slide_dir.rglob('*.txt'):
        # Ignora o classes.txt caso ele exista na origem
        if txt_file.name == 'classes.txt':
            continue
            
        teve_alteracao = False
        novas_linhas = []
        
        # Lê o arquivo
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Se encontrar a versão com "w" minúsculo, substitui
                if nome_errado in line:
                    line = line.replace(nome_errado, nome_correto)
                    teve_alteracao = True
                    linhas_corrigidas += 1
                novas_linhas.append(line)
                
        # Se encontrou e alterou alguma linha, sobrescreve o arquivo com a correção
        if teve_alteracao:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(novas_linhas)
            print(f"Corrigido: {txt_file.name}")
            arquivos_corrigidos += 1
            
    print(f"\nFeito! {linhas_corrigidas} rótulos foram corrigidos em {arquivos_corrigidos} arquivos.")


if __name__ == "__main__":
    corrigir_nomes_classes()