import os


def get_document_paths(directory: str = "./docs") -> list[str]:
    directory = os.path.abspath(directory)

    def join(f: str) -> str:
        return os.path.join(directory, f)

    return [join(f) for f in os.listdir(directory) if os.path.isfile(join(f))]
