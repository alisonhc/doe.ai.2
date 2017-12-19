from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from shutil import copyfileobj, move, rmtree
from tempfile import SpooledTemporaryFile
from re import sub, findall
import random
from collections import Counter

_dataDir = Path("data")
if not _dataDir.exists():
    _dataDir.mkdir()


def writeToFile(path, prompt, response, filename="train"):
    if not path.exists():
        path.mkdir()
    for lines, extension in [(prompt, "enc"), (response, "dec")]:
        with open(str(path / (filename + "." + extension)), "w") as f:
            f.write("\n".join(lines))


def makeTrainTest(*args, testPercent=0.1):
    l = len(args[0])
    for arg in args:
        assert l == len(arg)
    testIndices = set(random.sample(range(l), round(l * testPercent)))
    trainIndices = sorted(set(range(l)) - testIndices)
    testIndices = sorted(testIndices)
    return [[arg[i] for i in trainIndices] for arg in args], \
           [[arg[i] for i in testIndices] for arg in args]


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
    def maybeDownload(self, force=False):
        pass

    @abstractmethod
    def getData(self):
        pass


class _CornellMovieCorpus(_AbstractDataSource):
    def __init__(self):
        self.movieLines = self.localPath / "movie_lines.txt"
        self.movieConversations = self.localPath / "movie_conversations.txt"
        self.movieMeta = self.localPath / "movie_characters_metadata.txt"
        self.maybeDownload()

        with open(str(self.movieConversations), "r", errors="ignore") as f:
            self.conversations = [findall(r"L\d+", line.split("+++$+++")[-1]) for line in f]

        self._lineMap = {}
        with open(str(self.movieLines), "r", errors="ignore") as f:
            for line in f:
                _line = line.split("+++$+++")
                lineId = _line[0].strip()
                # get rid of newlines and beginning spaces
                self._lineMap[lineId] = _line[-1].strip()

        self._charMap = {}
        with open(str(self.movieMeta), errors="ignore") as f:
            for line in f:
                _line = line.split("+++$+++")
                name = _line[1].strip().lower()
                self._charMap[name] = _line[0].strip()

        self._prevLineMap = {}
        for lineIds in self.conversations:
            self._prevLineMap.update(zip(lineIds[1:], lineIds[:-1]))

    @property
    def localPath(self):
        return _dataDir / "cornell movie-dialogs corpus"

    @property
    def _url(self):
        return "http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

    def characterToId(self, character):
        if len(self._charMap) != 0:
            return self._charMap[character.lower()]

    def maybeDownload(self, force=False):
        if not self.localPath.exists() or force:
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

    def getMostCommonCharacters(self, num):
        with open(str(self.movieLines), errors="ignore") as f:
            names = [line.split("+++$+++")[3].strip().lower() for line in f]
        most_common = []
        most_common.extend(Counter(names).most_common(num))
        return [self.getCharacter(characterName=character[0]) for character in most_common], \
               [character[0] for character in most_common]

    def getCharacter(self, characterId=None, characterName=None):
        characterId = characterId or self.characterToId(characterName)
        prompt, response = [], []
        with open(str(self.movieLines), "r", errors="ignore") as f:
            for line in f:
                _line = line.split("+++$+++")
                lineId = _line[0].strip()
                if _line[1].strip() == characterId and lineId in self._prevLineMap:
                    prompt.append(self._lineMap[self._prevLineMap[lineId]])
                    response.append(self._lineMap[lineId])
        return prompt, response

    def makeYearSubset(self, movieYear):
        with open(str(self.localPath / "movie_titles_metadata.txt"), "r", errors="ignore") as f:
            year = str(movieYear)
            movie_ids = set()
            for line in f:
                if line.split("+++$+++")[2].strip()[:3] == year[:3]:
                    movie_ids.add(line.split("+++$+++")[0].strip())
        return movie_ids

    def makeYearFiles(self, movieYear):
        with open(str(self.localPath / "movie_lines.txt"), "r", errors="ignore") as f:
            yearConversations = []
            yearSubset = self.makeYearSubset(movieYear)
            for line in f:
                lineYear = line.split("+++$+++")[2].strip()
                if lineYear in yearSubset:
                    lineToAdd = line.split("+++$+++")[-1].strip()
                    yearConversations.append(lineToAdd)
        inputs = []
        outputs = []
        for i in range(len(yearConversations)):
            if i % 2 == 0:  # if i is even, it is an input. if i is odd, it is an output
                inputs.append(yearConversations[i])
            else:
                outputs.append(yearConversations[i])
        return inputs, outputs

    def getData(self):
        prompt = [self._lineMap[self._prevLineMap[lineId]] for lineId in self._prevLineMap]
        response = [self._lineMap[lineId] for lineId in self._prevLineMap]
        return prompt, response


class _UbuntuDialogCorpus(_AbstractDataSource):

    @property
    def _url(self):
        return None

    @property
    def localPath(self, **kwargs):
        return _dataDir / "ubuntu_dialog_corpus"

    def maybeDownload(self, force=False):
        if not self.localPath.exists():
            self.localPath.mkdir()
            raise FileExistsError("Please download the Ubuntu dialog corpus and follow the instructions here"
                                  "https://github.com/rkadlec/ubuntu-ranking-dataset-creator. Put the train / test "
                                  "valid files in the path {}".format(str(self.localPath)))

    def getData(self):
        self.maybeDownload()

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
        return prompt, response


class DataSource(Enum):
    CORNELL_MOVIE_CORPUS = _CornellMovieCorpus
    UBUNTU_DIALOG_CORPUS = _UbuntuDialogCorpus


if __name__ == "__main__":
    cornell = DataSource.CORNELL_MOVIE_CORPUS.value()
    """cPrompt, cResponse = cornell.getData()
    train, test = makeTrainTest(cPrompt, cResponse)
    writeToFile(cornell.localPath / "all", train[0], train[1], "train")
    writeToFile(cornell.localPath / "all", test[0], test[1], "test")"""

    year = "1940"
    yearInput, yearOutput = cornell.makeYearFiles(year)
    yearTrain, yearTest = makeTrainTest(yearInput, yearOutput)
    writeToFile(cornell.localPath / year, yearTrain[0], yearTrain[1], "train")
    writeToFile(cornell.localPath / year, yearTest[0], yearTest[1], "test")

    """characters, names_ = cornell.getMostCommonCharacters(5)
    for i,char in enumerate(characters):
        train, test = makeTrainTest(*char)
        writeToFile(cornell.localPath / names_[i], train[0], train[1], "train")
        writeToFile(cornell.localPath / names_[i], test[0], test[1], "test")"""

    # ubuntu = DataSource.UBUNTU_DIALOG_CORPUS.value()
    # uPrompt,uResponse = ubuntu.getData()
    # train, test = makeTrainTest(uPrompt,uResponse)
    # writeToFile(ubuntu.localPath, train[0], train[1], "train")
    # writeToFile(ubuntu.localPath, test[0], test[1], "test")