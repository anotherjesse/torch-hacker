import platform
import subprocess


last_apts = set()
def install_apts(deps):
    global last_apts

    deps = parse_str(deps)
    if len(deps) == 0 or deps == last_apts:
        print("no changes made in apt packages")
        return False

    if platform.system() == "Darwin":
        return install_darwin_apts(deps)

    print("installing apt deps", deps)

    subprocess.run(["apt-get", "update"], check=True)
    result = subprocess.run(
        ["apt-get", "install", "-y", *list(deps)], capture_output=True, text=True, check=True
    )
    last_apts = deps
    if "0 newly installed" in result.stdout and "0 upgraded" in result.stdout:
        print("No changes made in apt packages.")
        return False
    else:
        print("Changes made in apt packages.")
        return True


def parse_str(deps):
    if deps is None:
        return set()
    return set([d.strip() for d in deps.split(" ") if len(d.strip()) > 0])


last_pips = set()
def install_pips(deps):
    global last_pips

    deps = parse_str(deps)
    if len(deps) == 0 or deps == last_pips:
        print("no changes made in pip packages")
        return False
    
    print("installing pip deps", deps)
    result = subprocess.run(
        ["pip", "install", *list(deps)], capture_output=True, text=True, check=True
    )
    last_pips = deps
    if any(
        keyword in result.stdout
        for keyword in [
            "Successfully installed",
            "Successfully uninstalled",
            "Upgraded",
        ]
    ):
        print("Changes made in pip packages.")
        print("pip freeze:")
        subprocess.run(["pip", "freeze"], check=True)
        return True
    else:
        print("No changes made in pip packages.")
        return False

def install_darwin_apts(deps):
    global last_apts
    not_needed_on_macos = set(["libgl1-mesa-glx"])
    deps = deps - not_needed_on_macos

    if len(deps) == 0 or deps == last_apts:
        return False

    installed = subprocess.run(
        ["brew", "list"], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    
    installed = set([i.strip() for i in installed if len(i.strip()) > 0])

    to_install = list(deps - installed)
    if len(to_install) == 0:
        print("no changes made in apt packages")
        last_apts = deps
        return False
    
    print("installing apt deps", to_install)
    subprocess.run(["brew", "install", *to_install], check=True)

    last_apts = deps
    return True