import Modules.data.mid2name
import Modules.data.FB15K_237KBCD

dataSets = {
    "fb15k": {
        "trainPath": str(list(Modules.data.FB15K_237KBCD.__path__)[0])+ "/train.txt",
        "validPath": str(list(Modules.data.FB15K_237KBCD.__path__)[0])+ "/valid.txt",
        "testPath": str(list(Modules.data.FB15K_237KBCD.__path__)[0])+ "/test.txt",
    },
    "mid2name": {
        "mapPath": str(list(Modules.data.mid2name.__path__)[0])+"/mid2name.tsv",
    }
}