# tcc-modulacoes-awgn

Código-fonte da simulação usada no Trabalho de Conclusão de Curso (TCC) **"Análise comparativa do desempenho das modulações BPSK, QPSK, 16-QAM e 64-QAM em canal AWGN por meio de simulação computacional"**, do curso de Engenharia Elétrica do Centro Universitário IESB.

- **Autor:** Fernando Gabriel Corrêa
- **Orientador:** Prof. Odilon Francisco Pavón Duarte

## O que o código faz

O script `simulacao_modulacoes.py` estima, pelo método de Monte Carlo, a taxa de erro de bit (BER) das modulações BPSK, QPSK, 16-QAM e 64-QAM em canal com ruído aditivo branco gaussiano (AWGN). Para cada valor de *E*<sub>b</sub>/*N*<sub>0</sub>, ele:

1. gera bits aleatórios;
2. faz o mapeamento Gray;
3. modula os bits em símbolos com energia média normalizada (*E*<sub>s</sub> = 1);
4. soma o ruído AWGN;
5. demodula pelo símbolo mais próximo;
6. conta os bits errados e calcula a BER.

Os resultados simulados são comparados com as expressões teóricas de BER apresentadas no artigo (Seção 3.6).

## Parâmetros (Tabela 1 do artigo)

| Parâmetro | Valor |
|---|---|
| Modulações | BPSK, QPSK, 16-QAM, 64-QAM |
| Canal | AWGN |
| Faixa de *E*<sub>b</sub>/*N*<sub>0</sub> | 0 a 20 dB, passo de 1 dB |
| Bits por ponto de *E*<sub>b</sub>/*N*<sub>0</sub> | 2 × 10⁶ |
| Mapeamento | Código Gray |
| Energia média de símbolo | Normalizada (*E*<sub>s</sub> = 1) |
| Semente aleatória | Fixa (42) |

## Figuras geradas

- `figura5_ber.png`: BER em função de *E*<sub>b</sub>/*N*<sub>0</sub>, com curvas teóricas e pontos simulados (Figura 5 do artigo).
- `figura6_constelacoes.png`: constelações recebidas com *E*<sub>b</sub>/*N*<sub>0</sub> = 12 dB (Figura 6 do artigo).

O script também imprime o *E*<sub>b</sub>/*N*<sub>0</sub> necessário para BER = 10⁻⁵ em cada modulação (Tabela 2 do artigo) e a BER simulada a 12 dB.

## Como executar

Versões usadas: Python 3.13, NumPy 2.5, SciPy 1.18 e Matplotlib 3.11.

```bash
pip install -r requirements.txt
python simulacao_modulacoes.py
```

A execução leva poucos segundos. Como a semente é fixa, os resultados e as figuras são reproduzidos exatamente.
