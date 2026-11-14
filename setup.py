from setuptools import find_packages, setup


def _parse_requirements(path: str) -> list:
    """Read requirements.txt and strip comments / blank lines."""
    with open(path) as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#") and not line.startswith("-e")
        ]


setup(
    name="student-score-predictor",
    version="1.0.0",
    author="Love Jaiswal",
    author_email="Lovejaiswal881@gmail.com",
    description="End-to-end ML pipeline to predict student math scores",
    packages=find_packages(),
    install_requires=_parse_requirements("requirements.txt"),
    python_requires=">=3.8",
)
