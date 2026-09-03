# UnitMatch

UnitMatch matches electrophysiological units within and across recording
sessions using waveform and spatial features.

This repository contains the Python implementations:

- **UnitMatchPy** for probabilistic unit matching and manual review.
- **DeepUnitMatch** for deep-learning-based matching.

## Installation

Install the released package:

```bash
pip install UnitMatchPy
```

For development, clone the repository and install the package from its
subdirectory:

```bash
cd UnitMatchPy
pip install -e ".[full,notebooks]"
```

See [UnitMatchPy/README.md](UnitMatchPy/README.md) for requirements, GPU setup,
demo notebooks, and detailed usage.

## References

- [UnitMatch paper](https://www.nature.com/articles/s41592-024-02440-1)
- [DeepUnitMatch preprint](https://www.biorxiv.org/content/10.64898/2026.01.30.702777v1)
- [Original UnitMatch repository](https://github.com/EnnyvanBeest/UnitMatch)

Original UnitMatch was developed by Enny H. van Beest and Celian Bimbard.
UnitMatchPy was developed by Sam Dodgson. DeepUnitMatch was developed by
Wentao Qiu and Suyash Agarwal.

## License

See [LICENSE](LICENSE).
