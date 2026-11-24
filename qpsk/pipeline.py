"""在 AWGN 信道上仿真 Gray QPSK，导出 BER 数据和图表。"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "qpsk"
EB_N0_DB = np.arange(13)
BIT_ENERGY = 0.5  # Es = 1，每个符号携带两个比特
AMPLITUDE = np.sqrt(BIT_ENERGY)


def map_bits(bits: np.ndarray) -> np.ndarray:
    pairs = bits.reshape(-1, 2)
    i = 1.0 - 2.0 * pairs[:, 0]
    q = 1.0 - 2.0 * pairs[:, 1]
    return AMPLITUDE * (i + 1j * q)


def decide_bits(symbols: np.ndarray) -> np.ndarray:
    return np.column_stack((symbols.real < 0, symbols.imag < 0)).astype(np.uint8).ravel()


def add_awgn(symbols: np.ndarray, eb_n0_db: float, rng: np.random.Generator) -> np.ndarray:
    n0 = BIT_ENERGY / (10 ** (eb_n0_db / 10))
    sigma = np.sqrt(n0 / 2)
    noise_i = rng.normal(0, sigma, symbols.shape)
    noise_q = rng.normal(0, sigma, symbols.shape)
    return symbols + noise_i + 1j * noise_q


def theoretical_ber(eb_n0_db: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(np.sqrt(10 ** (eb_n0_db / 10)))


def save_ber_plot(path: Path, error_counts: np.ndarray, n_bits: int, upper_95: float) -> None:
    dense_db = np.linspace(EB_N0_DB[0], EB_N0_DB[-1], 241)
    observed_errors = error_counts > 0

    figure, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.semilogy(dense_db, theoretical_ber(dense_db), color="#1e4b82", label="Theory")
    ax.semilogy(
        EB_N0_DB[observed_errors],
        error_counts[observed_errors] / n_bits,
        "o",
        color="#ad452a",
        label="Monte Carlo",
    )
    if np.any(~observed_errors):
        ax.semilogy(
            EB_N0_DB[~observed_errors],
            np.full(np.count_nonzero(~observed_errors), upper_95),
            "v",
            color="#533581",
            label="Zero errors: 95% upper bound",
        )
    ax.set(
        xlabel=r"$E_b/N_0$ (dB)",
        ylabel="Bit error rate (BER)",
        title=f"Gray-coded QPSK over AWGN · {n_bits:,} bits per point",
        xticks=EB_N0_DB,
    )
    ax.grid(True, which="major", alpha=0.35)
    ax.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_constellation(path: Path, sample: np.ndarray, rng: np.random.Generator) -> None:
    ideal = map_bits(np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8))
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout="constrained")
    for ax, db in zip(axes, (0, 8), strict=True):
        received = add_awgn(sample, db, rng)
        ax.scatter(received.real, received.imag, s=7, alpha=0.35, label="Received")
        ax.scatter(ideal.real, ideal.imag, marker="x", s=75, color="#a12830", label="Ideal")
        ax.axhline(0, color="0.45", linewidth=0.8)
        ax.axvline(0, color="0.45", linewidth=0.8)
        ax.set(
            title=rf"$E_b/N_0$ = {db} dB",
            xlabel="In-phase (I)",
            ylabel="Quadrature (Q)",
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            aspect="equal",
        )
        ax.legend(loc="upper right")
    figure.suptitle("QPSK received constellation")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QPSK 复基带链路仿真与 BER 曲线")
    parser.add_argument(
        "--bits", type=int, default=400_000, help="每个信噪比的比特数（默认 400000）"
    )
    parser.add_argument("--seed", type=int, default=2026, help="随机种子（默认 2026）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="产物目录")
    args = parser.parse_args(argv)
    if args.bits <= 0 or args.bits % 2:
        parser.error("--bits 必须是正偶数")
    if args.seed < 0:
        parser.error("--seed 必须是非负整数")

    # 星座抽样单独用一条随机流，绘图参数不会改变 BER。
    simulation_seed, plot_seed = np.random.SeedSequence(args.seed).spawn(2)
    rng = np.random.default_rng(simulation_seed)
    bits = rng.integers(0, 2, size=args.bits, dtype=np.uint8)
    transmitted = map_bits(bits)
    error_counts = []
    for db in EB_N0_DB:
        received = add_awgn(transmitted, db, rng)
        recovered = decide_bits(received)
        error_counts.append(np.count_nonzero(recovered != bits))
    error_counts = np.array(error_counts)
    upper_95 = -np.expm1(np.log(0.05) / args.bits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "qpsk_ber.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "eb_n0_db",
                "total_bits",
                "bit_errors",
                "simulated_ber",
                "theory_ber",
                "zero_error_upper_95",
            ]
        )
        for db, errors, theory in zip(
            EB_N0_DB, error_counts, theoretical_ber(EB_N0_DB), strict=True
        ):
            writer.writerow(
                [db, args.bits, errors, errors / args.bits, theory, upper_95 if errors == 0 else ""]
            )

    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 11})
    save_ber_plot(args.output_dir / "qpsk_ber.png", error_counts, args.bits, upper_95)
    save_constellation(
        args.output_dir / "qpsk_constellation.png",
        transmitted[:1200],
        np.random.default_rng(plot_seed),
    )
    print(
        f"每点 {args.bits:,} 比特，随机种子 {args.seed}。已写入 BER 表和两张图：{args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
