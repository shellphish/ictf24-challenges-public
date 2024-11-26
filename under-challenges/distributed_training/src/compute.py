import torch
import sys
import os

flagfile = '/flag.txt'
if not os.path.exists(flagfile):
    FLAG = "ictf{not_easy_to_get_the_flag_x}"
else:
    with open(flagfile) as f:
        FLAG = f.read().strip()
MATRIX_SIZE = (4, 4)


def flag_to_matrix(flag: str) -> torch.Tensor:
    # Convert the flag to a matrix of integers
    assert len(flag) == 32
    # two characters per integer
    mat = torch.tensor(
        [ord(flag[i]) + (ord(flag[i + 1]) << 8) for i in range(0, len(flag), 2)],
        dtype=torch.float,
    )
    mat = mat.view(MATRIX_SIZE)

    return mat


def load_matrix_from_file(file_path: str) -> torch.Tensor:
    # Load the matrix from the file
    mat = torch.load(file_path)

    assert mat.size() == MATRIX_SIZE

    return mat


def main():
    mat_file = sys.argv[1]
    try:
        mat = load_matrix_from_file(mat_file)
        flag_mat = flag_to_matrix(FLAG)

        res = torch.matmul(mat, flag_mat)
        res_lst = res.tolist()
        print(res_lst)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
