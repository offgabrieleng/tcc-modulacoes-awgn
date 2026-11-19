"""
Simulação de Monte Carlo do desempenho das modulações BPSK, QPSK, 16-QAM e
64-QAM em canal AWGN.

TCC – Engenharia Elétrica (IESB)
Autor: Fernando Gabriel Corrêa

Gera:
  - Figura 5: BER em função de Eb/N0 (curvas teóricas e pontos simulados);
  - Figura 6: constelações recebidas com Eb/N0 = 12 dB.

Parâmetros (Tabela 1 do artigo):
  - Modulações: BPSK, QPSK, 16-QAM, 64-QAM
  - Canal: AWGN
  - Faixa de Eb/N0: 0 a 20 dB, passo de 1 dB
  - Bits por ponto de Eb/N0: 2 x 10^6
  - Mapeamento: código Gray
  - Energia média de símbolo: normalizada (Es = 1)
  - Semente aleatória: fixa (42)

Requisitos: Python 3, NumPy, SciPy e Matplotlib.
Execução:   python simulacao_modulacoes.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Parâmetros da simulação (Tabela 1)
# ---------------------------------------------------------------------------
SEMENTE = 42
BITS_POR_PONTO = 2_000_000
EBN0_DB = np.arange(0, 21, 1)          # 0 a 20 dB, passo de 1 dB
EBN0_CONSTELACAO_DB = 12               # Eb/N0 usado na Figura 6
SIMBOLOS_CONSTELACAO = 2000            # símbolos plotados por constelação

# nome: bits por símbolo (k = log2 M)
MODULACOES = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}
CORES = {"BPSK": "C0", "QPSK": "C1", "16-QAM": "C2", "64-QAM": "C3"}

rng = np.random.default_rng(SEMENTE)


# ---------------------------------------------------------------------------
# Código Gray
# ---------------------------------------------------------------------------
def binario_para_gray(n):
    """Converte um inteiro (ou vetor de inteiros) para código Gray."""
    return n ^ (n >> 1)


def gray_para_binario(g):
    """Converte código Gray de volta para binário natural."""
    n = g.copy()
    desloc = g >> 1
    while np.any(desloc):
        n ^= desloc
        desloc >>= 1
    return n


def bits_para_inteiros(bits, b):
    """Agrupa os bits em blocos de b bits e converte cada bloco em inteiro."""
    pesos = 1 << np.arange(b - 1, -1, -1)
    return bits.reshape(-1, b) @ pesos


def inteiros_para_bits(valores, b):
    """Converte inteiros de volta em blocos de b bits."""
    desloc = np.arange(b - 1, -1, -1)
    return ((valores[:, None] >> desloc) & 1).astype(np.int8).ravel()


# ---------------------------------------------------------------------------
# Modulação e demodulação
# ---------------------------------------------------------------------------
# A M-QAM quadrada é tratada como duas PAM com L = sqrt(M) níveis, uma no eixo
# em fase (I) e outra no eixo em quadratura (Q). Em cada eixo, o código Gray
# garante que níveis vizinhos diferem em apenas 1 bit. A QPSK é o caso M = 4.
# A BPSK usa apenas o eixo I, com os símbolos -1 e +1.

def fator_normalizacao(k):
    """Fator que torna a energia média de símbolo igual a 1 (Es = 1)."""
    if k == 1:
        return 1.0
    M = 2 ** k
    return np.sqrt(2 * (M - 1) / 3)


def modular(bits, k):
    """Mapeia a sequência de bits em símbolos complexos com Es = 1."""
    if k == 1:                                    # BPSK: 0 -> -1, 1 -> +1
        return (2 * bits - 1).astype(float) + 0j
    b = k // 2                                    # bits por eixo
    L = 2 ** b                                    # níveis por eixo
    grupos = bits.reshape(-1, k)
    gi = bits_para_inteiros(grupos[:, :b].ravel(), b)
    gq = bits_para_inteiros(grupos[:, b:].ravel(), b)
    ni = gray_para_binario(gi)                    # índice do nível (0..L-1)
    nq = gray_para_binario(gq)
    nivel_i = 2 * ni - (L - 1)                    # níveis ±1, ±3, ...
    nivel_q = 2 * nq - (L - 1)
    return (nivel_i + 1j * nivel_q) / fator_normalizacao(k)


def demodular(r, k):
    """Decide o símbolo mais próximo em cada eixo e devolve os bits."""
    if k == 1:
        return (r.real > 0).astype(np.int8)
    b = k // 2
    L = 2 ** b
    escala = fator_normalizacao(k)
    ni = np.clip(np.rint((r.real * escala + (L - 1)) / 2), 0, L - 1).astype(int)
    nq = np.clip(np.rint((r.imag * escala + (L - 1)) / 2), 0, L - 1).astype(int)
    bits_i = inteiros_para_bits(binario_para_gray(ni), b).reshape(-1, b)
    bits_q = inteiros_para_bits(binario_para_gray(nq), b).reshape(-1, b)
    return np.hstack([bits_i, bits_q]).ravel()


def canal_awgn(s, k, ebn0_db):
    """Soma ruído gaussiano branco com a densidade N0 correspondente a Eb/N0."""
    ebn0 = 10 ** (ebn0_db / 10)
    eb = 1.0 / k                                  # Es = 1  ->  Eb = Es / k
    n0 = eb / ebn0
    sigma = np.sqrt(n0 / 2)                       # variância N0/2 por dimensão
    if k == 1:                                    # BPSK: sinal real
        return s + sigma * rng.standard_normal(s.size)
    ruido = rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size)
    return s + sigma * ruido


# ---------------------------------------------------------------------------
# Expressões teóricas (Seção 3.6)
# ---------------------------------------------------------------------------
def funcao_q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def ber_teorica(k, ebn0_db):
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    if k <= 2:                                    # BPSK e QPSK
        return funcao_q(np.sqrt(2 * ebn0))
    M = 2 ** k                                    # M-QAM quadrada com Gray
    return (4 / k) * (1 - 1 / np.sqrt(M)) * funcao_q(
        np.sqrt(3 * k / (M - 1) * ebn0))


# ---------------------------------------------------------------------------
# Simulação de Monte Carlo
# ---------------------------------------------------------------------------
def simular_ber(k):
    """Estima a BER para cada valor de Eb/N0: bits errados / bits enviados."""
    n_bits = (BITS_POR_PONTO // k) * k            # múltiplo de k
    ber = []
    for ebn0_db in EBN0_DB:
        bits = rng.integers(0, 2, n_bits, dtype=np.int8)
        r = canal_awgn(modular(bits, k), k, ebn0_db)
        erros = np.count_nonzero(demodular(r, k) != bits)
        ber.append(erros / n_bits)
    return np.array(ber)


def main():
    resultados = {}
    for nome, k in MODULACOES.items():
        print(f"Simulando {nome}...")
        resultados[nome] = simular_ber(k)

    # ----- Figura 5: BER x Eb/N0 -------------------------------------------
    ebn0_fino = np.linspace(0, 20, 401)
    fig, ax = plt.subplots(figsize=(8, 6))
    for nome, k in MODULACOES.items():
        ax.semilogy(ebn0_fino, ber_teorica(k, ebn0_fino), "-",
                    color=CORES[nome], label=f"{nome} (teórica)")
        ber_sim = np.where(resultados[nome] > 0, resultados[nome], np.nan)
        ax.semilogy(EBN0_DB, ber_sim, "o", color=CORES[nome],
                    label=f"{nome} (simulada)")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("Taxa de erro de bit (BER)")
    ax.set_title("Desempenho de modulações digitais em canal AWGN")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig("figura5_ber.png", dpi=300)
    plt.close(fig)

    # ----- Figura 6: constelações recebidas com Eb/N0 = 12 dB --------------
    fig, eixos = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (nome, k) in zip(eixos, MODULACOES.items()):
        bits = rng.integers(0, 2, SIMBOLOS_CONSTELACAO * k, dtype=np.int8)
        r = canal_awgn(modular(bits, k), k, EBN0_CONSTELACAO_DB)
        ax.plot(r.real, r.imag, ".", markersize=2, alpha=0.5, color=CORES[nome])
        ideais = np.unique(modular(np.array(
            [int(c) for i in range(2 ** k) for c in format(i, f"0{k}b")],
            dtype=np.int8), k))
        ax.plot(ideais.real, ideais.imag, "k+", markersize=9, markeredgewidth=1.5)
        ax.set_title(f"{nome} — $E_b/N_0$ = {EBN0_CONSTELACAO_DB} dB")
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.75, 1.75)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_xlabel("Em fase (I)")
    eixos[0].set_ylabel("Quadratura (Q)")
    fig.tight_layout()
    fig.savefig("figura6_constelacoes.png", dpi=300)
    plt.close(fig)

    # ----- Resumo numérico (Tabela 2 e Seções 4.2 a 4.4) -------------------
    print("\nEb/N0 necessário para BER = 1e-5 (curva teórica):")
    for nome, k in MODULACOES.items():
        ber_t = ber_teorica(k, ebn0_fino)
        valor = np.interp(-5, np.log10(ber_t[::-1]), ebn0_fino[::-1])
        print(f"  {nome:7s}: {valor:5.1f} dB")

    print(f"\nBER simulada com Eb/N0 = {EBN0_CONSTELACAO_DB} dB:")
    i12 = int(np.where(EBN0_DB == EBN0_CONSTELACAO_DB)[0][0])
    for nome in MODULACOES:
        print(f"  {nome:7s}: {resultados[nome][i12]:.2e}")

    print("\nFiguras salvas: figura5_ber.png e figura6_constelacoes.png")


if __name__ == "__main__":
    main()
