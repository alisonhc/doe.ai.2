from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from shutil import copyfileobj, move, rmtree
from tempfile import SpooledTemporaryFile
from re import sub, findall

_dataDir = Path("data")
if not _dataDir.exists():
    _dataDir.mkdir()


class _AbstractDataSource(ABC):
    @property
    @abstractmethod
    def _url(self):
        return ""

    @property
    @abstractmethod
    def localPath(self, **kwargs):
        return ""

    @abstractmethod
    def getData(self):
        pass


class _CornellMovieCorpus(_AbstractDataSource):
    @property
    def localPath(self):
        return _dataDir / "cornell movie-dialogs corpus"

    @property
    def _url(self):
        return "http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

    def characterToId(self, character):
        charId = ""
        with open(self.localPath / "movie_characters_metadata.txt", errors="ignore") as f:
            for line in f:
                _line = line.split("+++$+++")
                if _line[1].strip().lower() == character.lower():
                    charId = _line[0].strip()
                    break
        return charId

    def getData(self, character=None, characterId=None):
        if not self.localPath.exists():
            rootZipDir = "cornell movie-dialogs corpus"
            with urlopen(self._url) as response, SpooledTemporaryFile() as tmp:
                copyfileobj(response, tmp)
                with ZipFile(tmp) as zipTmp:
                    infoList = zipTmp.infolist()
                    for info in infoList:
                        pathFile = Path(info.filename)
                        if pathFile.parts[0] == rootZipDir and pathFile.stem != ".DS_Store":
                            zipTmp.extract(info, self.localPath)
            for p in self.localPath.joinpath(rootZipDir).iterdir():
                move(str(p), str(self.localPath))
            rmtree(str(self.localPath / rootZipDir))

        # get first character id with name of character
        charId = characterId or self.characterToId(character)

        with open(self.localPath / "movie_conversations.txt", "r", errors="ignore") as f:
            conversations = [findall(r"\d+", line.split("+++$+++")[-1]) for line in f]

        # for some reason, python can't parse the file at all without errors="ignore", something about uft-8 encoding
        with open(self.localPath / "movie_lines.txt", "r", errors="ignore") as f:
            lineDict = {}
            for line in f:
                _line = line.split("+++$+++")
                lineId = _line[0][1:].strip()
                lineCharacterId = _line[1].strip()
                lineDict[lineId] = (lineCharacterId, _line[-1].strip())  # get rid of newlines and beginning spaces

        characterConversations = []
        for conversation in conversations:
            for i in range(len(conversation)-1):
                if lineDict[conversation[i+1]][0] == charId:
                    prompt = lineDict[conversation[i]][1]
                    response = lineDict[conversation[i+1]][1]
                    characterConversations.append([prompt, response])

        conversationPath = self.localPath / charId
        if not conversationPath.exists():
            conversationPath.mkdir()
        inputFile = conversationPath / "train.enc"
        outputFile = conversationPath / "train.dec"
        inputs = [c[:-1] for c in characterConversations if len(c) >= 2]
        outputs = [c[1:] for c in characterConversations if len(c) >= 2]
        with open(inputFile, "w") as f:
            f.write("\n".join([line for c in inputs for line in c]))
        with open(outputFile, "w") as f:
            f.write("\n".join([line for c in outputs for line in c]))

        return characterConversations, inputFile, outputFile


class _UbuntuDialogCorpus(_AbstractDataSource):

    @property
    def _url(self):
        return None

    @property
    def localPath(self, **kwargs):
        return _dataDir / "ubuntu_dialog_corpus"

    def getData(self):
        if not self.localPath.exists():
            self.localPath.mkdir()
            raise FileExistsError("Please download the Ubuntu dialog corpus and follow the instructions here"
                                  "https://github.com/rkadlec/ubuntu-ranking-dataset-creator. Put the train / test "
                                  "valid files in the path {}".format(str(self.localPath)))
        # build training prompt / response pairs
        prompt, response = [], []
        trainFile = self.localPath / "train.csv"
        with open(str(trainFile), "r") as f:
            f.readline()  # get rid of first line
            for line in f:
                parts = line.split("__eot__")
                if float(line.split(",")[-1].strip()) == 1.0:  # if the line is labeled as correct
                    tmpResponse = parts[-1][:-5].lower().strip(' ,"')
                    if len(findall(r"__eou__", tmpResponse)) < 2:
                        response.append(sub(r"\s?__eou__\s?", r" ", tmpResponse))
                        tmpPrompt = parts[-2].lower().strip()
                        prompt.append(sub(r"\s?__eou__\s?", r" ", tmpPrompt))

        with open(str(self.localPath / "train.enc"), "w") as f:
            f.write("\n".join(prompt))

        with open(str(self.localPath / "train.dec"), "w") as f:
            f.write("\n".join(response))

        return prompt, response


class DataSource(Enum):
    CORNELL_MOVIE_CORPUS = _CornellMovieCorpus()
    UBUNTU_DIALOG_CORPUS = _UbuntuDialogCorpus()


if __name__ == "__main__":
    ubuntu = DataSource.UBUNTU_DIALOG_CORPUS.value.getData()
