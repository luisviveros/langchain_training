import subprocess

with open("requirements.txt") as f:
    for line in f:
        pkg = line.strip()
        if pkg and not pkg.startswith("#"):
            print(f"Instalando {pkg} desde PyPI...")
            subprocess.run(
                ["pip", "install", pkg, "--index-url", "https://pypi.org/simple"],
                check=False
            )