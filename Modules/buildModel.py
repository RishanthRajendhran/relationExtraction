from Modules.helper.imports.functionImports import checkFile, createDataLoader, trainModel, evaluateModel
from Modules.helper.imports.packageImports import argparse, pickle, logging, np, tf, hub, text, plt, train_test_split, transformers, sns, pd, torch, preprocessing, nltk, resample
from Modules.helper.imports.classImports import RelExtDataset, RelationClassifier

parser = argparse.ArgumentParser()

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to enable debug mode"
)

parser.add_argument(
    "-log",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-train",
    help="Path to file containing sampled training examples (extension=.pkl)",
    default="preprocessedExamples_17.pkl"
)

parser.add_argument(
    "-valid",
    help="Path to file containing sampled validation examples (extension=.pkl)",
    default="preprocessedExamples_valid_17.pkl"
)

parser.add_argument(
    "-test",
    help="Path to file containing sampled test examples (extension=.pkl)",
    default="preprocessedExamples_test_17.pkl"
)

parser.add_argument(
    "-trainValTest",
    action="store_true",
    help="Boolean flag to split train set into train, validation and test set",
    default=False
)

parser.add_argument(
    "-histogram",
    action="store_true",
    help="Boolean flag to show histogram of examples"
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size for dataloader",
    default=64
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for training",
    default=0.0001
)

parser.add_argument(
    "-epochs",
    type=int,
    help="No. of epochs to train for",
    default=10
)

parser.add_argument(
    "-load",
    help="Path to file containing model to load",
    default=None
)

parser.add_argument(
    "-maxSents",
    type=int,
    help="Maximum no. of sentences per examples",
    default=5
)

parser.add_argument(
    "-embeddingSize",
    type=int,
    help="Size of embeddings in sentence vectors (from Spacy)",
    default=300
)

parser.add_argument(
    "-entTypeToInd",
    help="Path to .pkl file containg mapping between NER types and integers",
    default="entTypeToInd.pkl"
)

parser.add_argument(
    "-posToInd",
    help="Path to .pkl file containg mapping between POS tags and integers",
    default="posToInd.pkl"
)

parser.add_argument(
    "-lemmaToInd",
    help="Path to .pkl file containg mapping between lemma and integers",
    default="lemmaToInd.pkl"
)

parser.add_argument(
    "-windowSize",
    type=int,
    help="Size of context window to consider",
    default=3
)

parser.add_argument(
    "-hiddenSize",
    type=int,
    help="Size of hidden representation in TreeLSTM",
    default=1024
)

parser.add_argument(
    "-dropout",
    type=float,
    help="Dropout to be applied in TreeLSTM",
    default=0.3
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
trainFile = args.train
validFile = args.valid
testFile = args.test
trainValTest = args.trainValTest
histogram = args.histogram
batchSize = args.batchSize
epochs = args.epochs
loadModel = args.load
learningRate = args.learningRate
maxSents = args.maxSents
embeddingSize = args.embeddingSize
entTypeToIndFile = args.entTypeToInd
posToIndFile = args.posToInd
lemmaToIndFile = args.lemmaToInd
windowSize = args.windowSize
hiddenSize = args.hiddenSize
dropout = args.dropout

if logFile:
    checkFile(logFile, ".txt")
checkFile(trainFile, ".pkl")
checkFile(validFile, ".pkl")
checkFile(testFile, ".pkl")
checkFile(entTypeToIndFile, ".pkl")
checkFile(posToIndFile, ".pkl")
checkFile(lemmaToIndFile, ".pkl")

if loadModel:
    checkFile(loadModel, ".pt")

with open(entTypeToIndFile, "rb") as f:
    entTypeToInd = pickle.load(f)

with open(posToIndFile, "rb") as f:
    posToInd = pickle.load(f)

with open(lemmaToIndFile, "rb") as f:
    lemmaToInd = pickle.load(f)

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

with open(trainFile, "rb") as f:
    train = pickle.load(f)

if not trainValTest:
    with open(validFile, "rb") as f:
        valid = pickle.load(f)
    with open(testFile, "rb") as f:
        test = pickle.load(f)

df = pd.DataFrame(train)
if histogram:
    vals, counts = np.unique(df["relation"].to_numpy(),return_counts=True)  
    vals = [v.split("/")[-1] for v in vals]
    plt.bar(vals, counts)
    plt.xlabel("Relation")
    plt.ylabel("No. of instances")
    plt.title("No. of instances in train set per relation")
    plt.show()
    plt.clf()
    relNames = df["relation"].unique().tolist()
    relSentCounts = {}
    for rel in relNames:
        relSentCounts[rel.split("/")[-1]] = 0
        dfRel = df[df["relation"] == rel]
        for _, row in dfRel.iterrows():
            relSentCounts[rel.split("/")[-1]] += len(row["texts"])
        relSentCounts[rel.split("/")[-1]] /= len(dfRel)
        relSentCounts[rel.split("/")[-1]] = round(relSentCounts[rel.split("/")[-1]], 2)
    plt.bar(relSentCounts.keys(), relSentCounts.values())
    plt.xlabel("Relation")
    plt.ylabel("Avg. sentences per example instance")
    plt.title("Avg. sentences per example instance per relation")
    plt.show()
    plt.clf()

#Bert model expects integer targets, not strings
if not loadModel:
    le = preprocessing.LabelEncoder()
    df['relation'] = le.fit_transform(df['relation'])
    with open("labelEncoder.pkl","wb") as f:
        pickle.dump(le, f)

#Balance dataset 
vals, counts = np.unique(df.relation.to_numpy(), return_counts=True)
if not np.allclose(counts, [(sum(counts)/len(counts))]*len(vals)):
    logging.info("Dataframe not balanced!")
    majorVal = np.argmax(counts)
    dfMajor = df[df.relation == vals[majorVal]]
    numSamples = dfMajor.shape[0]
    for val in vals: 
        if val == vals[majorVal]:
            continue
        dfTemp = df[df.relation == val]
        dfTemp = resample(
            dfTemp, 
            replace=True, 
            n_samples=numSamples,
            random_state=26
        )
        dfMajor = pd.concat([dfMajor, dfTemp])
    df = dfMajor.copy()
    logging.info(f"Resampled dataframe to balance it")
    vals, counts = np.unique(df.relation.to_numpy(), return_counts=True)
    logging.info(f"vals: {vals}")
    logging.info(f"new counts: {counts}")
#Balance dataset

if not trainValTest:
    df_train = df
    df_valid = pd.DataFrame(valid)
    df_test = pd.DataFrame(test)
    if not loadModel:
        df_valid['relation'] = le.transform(df_valid['relation'])
        df_test['relation'] = le.transform(df_test['relation'])
else:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
    df_valid, df_test = train_test_split(df_test, test_size=0.6, random_state=13)

classes = np.unique(df.relation.to_numpy())

if loadModel:
    model = torch.load(loadModel)
    if not trainValTest:
        df_train["relation"] = model.textToLabel(df_train["relation"])
        df_valid["relation"] = model.textToLabel(df_valid["relation"])
        df_test["relation"] = model.textToLabel(df_test["relation"])
    else: 
        df["relation"] = model.textToLabel(df["relation"])
else:
    model = RelationClassifier(len(classes), maxSents, embeddingSize, hiddenSize, windowSize, dropout, le)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
model = model.to(device)
 
trainDataLoader = createDataLoader(df_train, embeddingSize, entTypeToInd, posToInd, lemmaToInd, maxSents, hiddenSize, windowSize, batchSize, device)
validDataLoader = createDataLoader(df_valid, embeddingSize, entTypeToInd, posToInd, lemmaToInd, maxSents, hiddenSize, windowSize, batchSize, device)
testDataLoader = createDataLoader(df_test, embeddingSize, entTypeToInd, posToInd, lemmaToInd, maxSents, hiddenSize, windowSize, batchSize, device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=learningRate, 
)

totalSteps = epochs
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=totalSteps
)
lossFunction = torch.nn.CrossEntropyLoss().to(device)

history = {
    "trainAcc": [],
    "trainLoss": [],
    "validAcc": [],
    "validLoss": [],
    "testAcc": [],
    "testLoss": []
}
bestAcc = 0 

for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs}")

    trainAcc, trainLoss = trainModel(
        model,
        trainDataLoader,
        lossFunction,
        optimizer, 
        device, 
        scheduler,
        len(df_train)
    )

    logging.info(f"\tTrain Accuracy: {trainAcc}, Train loss: {trainLoss}")

    validAcc, validLoss = evaluateModel(
        model,
        validDataLoader,
        lossFunction,
        device, 
        len(df_valid)
    )

    logging.info(f"\tValidation Accuracy: {validAcc}, Validation loss: {validLoss}")

    testAcc, testLoss = evaluateModel(
        model,
        testDataLoader,
        lossFunction,
        device, 
        len(df_test)
    )

    logging.info(f"\tTest Accuracy: {testAcc}, Tess loss: {testLoss}")

    history["trainAcc"].append(trainAcc)
    history["trainLoss"].append(trainLoss)

    history["validAcc"].append(validAcc)
    history["validLoss"].append(validLoss)

    history["testAcc"].append(validAcc)
    history["testLoss"].append(validLoss)

    if validAcc > bestAcc: 
        torch.save(model, "fullModel.pt")
        torch.save(model.state_dict(), "modelStateDict.pt")
        bestAcc = validAcc

with open("history.pkl","wb") as f:
    pickle.dump(history, f)