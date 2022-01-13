import sys


def find_line(file, value):
    """Find line in file that contains number closest to value.
        Assuming the lines are sorted.
    """
    diff = 1e4  # big number
    for i, line in enumerate(file):
        line_value = float(line.strip())
        new_diff = abs(value - line_value)
        if new_diff < diff:
            diff = new_diff
        elif new_diff > diff:
            i -= 1
            break
    return i


def main():
    if len(sys.argv) != 3:
        print("Error need to specify a simulation directory and desired output time")
        sys.exit(1)

    outputs_file = sys.argv[1] + "/outputs.txt"
    output_time = float(sys.argv[2])

    with open(outputs_file, 'r') as f:
        print(f"{find_line(f, output_time):03}")


if __name__ == "__main__":
    main()
