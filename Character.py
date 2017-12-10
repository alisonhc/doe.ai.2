from DataSource import DataSource
from collections import Counter


class Character:
    def __init__(self):
        self.cornell_data = DataSource.CORNELL_MOVIE_CORPUS.value

    def most_common(self, num=5):
        with open(str(self.cornell_data.localPath / "movie_lines.txt"), errors="ignore") as f:
            char_ids = [line.split("+++$+++")[1].strip() for line in f]
        most_common = []
        most_common.extend(Counter(char_ids).most_common(num))
        return [self.cornell_data.getData(characterId=character[0]) for character in most_common]


if __name__ == "__main__":
    characterMaker = Character()
    characterMaker.most_common(5)