import os
from sgfmill import sgf
import re
import shutil
import tarfile
import sys

def extract(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()


def filterGame(gameFile, rank):
    with open(gameFile, "rb") as g:
        game = sgf.Sgf_game.from_bytes(g.read())
    root_node = game.get_root()
    b_player = root_node.get("BR")
    w_player = root_node.get("WR")
    b_re = re.match('^\d+$', b_player)
    w_re = re.match('^\d+$', w_player)

    if not(b_re):
        #print("Drop game "+gameFile+" due to incorrect ranking")
        g.close()
        return False
    if not(w_re):
        #print("Drop game "+gameFile+" due to incorrect ranking")
        g.close()
        return False
    if((int(b_player) < rank) & (int(w_player) < rank)):
        g.close()
        return False
    g.close()
    return True


def filter_archive(archive, src, dst, rank=1800):
    print("Processing archive "+ archive)
    extract(src+'/'+archive)
    subdirname = os.path.splitext(archive)[0]
    if subdirname.endswith("tar"):
        subdirname = os.path.splitext(subdirname)[0]
    allGames = os.listdir(subdirname)
    print("Archive contains "+str(len(allGames))+" games")
    filterDir = dst + '/'+ subdirname + '-' + str(rank)
    if not os.path.isdir(filterDir):
        os.mkdir(filterDir)
    count = 0
    for game in allGames:
        res = filterGame(subdirname + '/' + game, rank)
        if res:
            shutil.copy(subdirname + '/' + game, filterDir)
            count += 1
    shutil.rmtree(subdirname)
    with tarfile.open(filterDir + '.tar.gz', "w:gz") as tar:
        tar.add(filterDir)
        tar.close()
    print("Archive processed, "+ str(count)+" valid games above the threshold of "+str(rank))
    shutil.rmtree(filterDir)




if __name__ == "__main__":

    #setting default values
    rank = 1500;
    src = 'games'
    dst = 'null'
    mode = 0
    argc = len(sys.argv)
    for i in range(1,argc):
        if sys.argv[i] == "-help":
            mode = 0;
        if sys.argv[i] == '-d':
            mode = 1;
            if i+1 < argc:
                dst = sys.argv[i+1]
        if sys.argv[i] == '-s':
            mode = 1;
            if i+1 < argc:
                src = sys.argv[i+1]
        if sys.argv[i] == '-f':
            if i+1 < argc:
                rank = int(sys.argv[i+1])

    if mode == 0:
        print("Usage :")
        print("-s <dirName> to choose the source (by default will check /games in the local directory)")
        print("-d <dirName> to choose the destination (by default will create a filtered folder in the source directory)")
        print("-f <rank> to set the rank threshold (1500 by default)")
    elif mode == 1:
        if not os.path.isdir(src):
            print("Error, no source directory found")
            exit()
        if dst == 'null':
            dst = src+"/filtered"
        archives = os.listdir(src)
        if not os.path.isdir(dst):
            os.mkdir(dst);
        print("Starting to process the archives in /"+src)
        print("Result will be stored in /" +dst)
        print("Ranking threshhold is "+str(rank))
        for f in archives:
            if (f.endswith("tar.gz") | f.endswith("tar")):
                filter_archive(f, src, dst, rank)
        print('All archives have been processed!')

    else:
        print("Unknown mode")




