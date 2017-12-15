from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from shutil import copyfileobj, move, rmtree
from tempfile import SpooledTemporaryFile
from re import sub, findall
import random
import string

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
        with open(str(self.localPath / "movie_characters_metadata.txt"), errors="ignore") as f:
            for line in f:
                _line = line.split("+++$+++")
                if _line[1].strip().lower() == character.lower():
                    charId = _line[0].strip()
                    break
        return charId

    def getConversations(self):
        with open(str(self.localPath / "movie_conversations.txt"), "r", errors="ignore") as f:
            conversations = [findall(r"\d+", line.split("+++$+++")[-1]) for line in f]
        return conversations

    def addToLineDict(self, line, lineDict):
        _line = line.split("+++$+++")
        lineId = _line[0][1:].strip()
        lineCharacterId = _line[1].strip()
        lineDict[lineId] = (lineCharacterId, _line[-1].strip())
        return lineDict

    def getCharacterConversations(self, conversations, lineDict, charId):
        characterConversations = []
        for conversation in conversations:
            for i in range(len(conversation)-1):
                if lineDict[conversation[i+1]][0] == charId:
                    prompt = lineDict[conversation[i]][1]
                    response = lineDict[conversation[i+1]][1]
                    characterConversations.append([prompt, response])
        return characterConversations

    def makeYearSubset(self, movieYear):
        with open(str(self.localPath / "movie_titles_metadata.txt"), "r", errors="ignore") as f:
            year = str(movieYear)
            movie_ids = set()
            for line in f:
                if line.split("+++$+++")[2].strip()[:3] == year[:3]:
                    movie_ids.add(line.split("+++$+++")[0].strip())
        return movie_ids

    def getData(self, character=None, characterId=None, movieYear=None):
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


        # for some reason, python can't parse the file at all without errors="ignore", something about uft-8 encoding
        with open(str(self.localPath / "movie_lines.txt"), "r", errors="ignore") as f:
            lineDict = {}
            yearConversations = []
            yearSubset = self.makeYearSubset(movieYear)
            for line in f:
                lineYear = line.split("+++$+++")[2].strip()
                if lineYear in yearSubset:
                    lineToAdd = line.split("+++$+++")[-1].strip()
                    yearConversations.append(lineToAdd)
                lineDict = self.addToLineDict(line, lineDict)

        conversations = self.getConversations()
        conversationPath = ""
        if character:
            charId = characterId or self.characterToId(character)
            characterConversations = self.getCharacterConversations(conversations, lineDict, charId)
            conversationPath = self.localPath / charId
            inputs = [c[:-1] for c in characterConversations if len(c) >= 2]
            outputs = [c[1:] for c in characterConversations if len(c) >= 2]

        elif movieYear:
            conversationPath = self.localPath / str(movieYear)
            inputs = []
            outputs = []
            for i in range(len(yearConversations)):
                if i%2==0:  # if i is even, it is an input. if i is odd, it is an output
                    inputs.append(yearConversations[i])
                else:
                    outputs.append(yearConversations[i])

        if not conversationPath.exists():
            conversationPath.mkdir()

        inputTrainFile = conversationPath / "train.enc"
        outputTrainFile = conversationPath / "train.dec"
        inputTestFile = conversationPath / "test.enc"
        outputTestFile = conversationPath / "test.dec"

        train_enc = open(str(inputTrainFile), "w")
        train_dec = open(str(outputTrainFile), "w")
        test_enc = open(str(inputTestFile), "w")
        test_dec = open(str(outputTestFile), "w")

        toSubtract = len(inputs)%2
        test_ids = set(random.sample([i for i in range(len(inputs)-toSubtract)], int((len(inputs))/2)-toSubtract))  # todo: is this a list or a set?Y
        for i in range(len(inputs)-toSubtract):
            if i in test_ids:
                #test_enc.write("\n".join([line for line in inputs]) + "\n")
                #test_dec.write("\n".join([line for line in outputs]) + "\n")
                test_enc.write(inputs[i] + "\n")
                test_dec.write(outputs[i] + "\n")
            else:
                #train_enc.write("\n".join([line for line in inputs]) + "\n")
                #train_dec.write("\n".join([line for line in outputs]) + "\n")
                train_enc.write(inputs[i] + "\n")
                train_dec.write(outputs[i] + "\n")
        train_enc.close()
        train_dec.close()
        test_enc.close()
        test_dec.close()
        return inputTrainFile, inputTestFile, outputTrainFile, outputTestFile


class DataSource(Enum):
    CORNELL_MOVIE_CORPUS = _CornellMovieCorpus()
    CORNELL_MOVIE_QUOTES_CORPUS = "https://www.cs.cornell.edu/~cristian/memorability_files/cornell_movie_quotes_corpus.zip"


if __name__ == "__main__":
    #cornell = DataSource.CORNELL_MOVIE_CORPUS.value.getData(character="Bianca")
    cornell = DataSource.CORNELL_MOVIE_CORPUS.value.getData(movieYear="1970")