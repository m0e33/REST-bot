if __name__ == "__main__":
    new_f = open("requirements_without_versions.txt", "w")
    with open("requirements.txt", "r") as f_old:
        for line in f_old:
            print(line)
            new_line = line.split("==")[0]
            if len(new_line) == len(line):
                # no split happened
                new_line = line.split("~=")[0]
            print(f"Writing {new_line} to file")
            new_f.write(new_line + "\n")
