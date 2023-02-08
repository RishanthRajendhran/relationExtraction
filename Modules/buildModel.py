from Modules.helper.imports.functionImports import checkFile, extractPOStags, findWordInWords, extractWordsInBetween, extractWordsInWindow, extractNERtags, getShortestDependencyPath, createDataLoader
from Modules.helper.imports.packageImports import argparse, pickle, logging, np, tf, hub, text, plt, train_test_split, transformers, sns, pd, torch, preprocessing
from Modules.helper.imports.classImports import RelExtDataset, RelationClassifier

tf.get_logger().setLevel("ERROR")

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
    help="Path to file containing training examples (extension=.pkl)",
    default="all_examples_2.pkl"
)

parser.add_argument(
    "-valid",
    help="Path to file containing validation examples (extension=.pkl)",
    default="all_examples_valid_2.pkl"
)

parser.add_argument(
    "-test",
    help="Path to file containing test  examples (extension=.pkl)",
    default="all_examples_test_2.pkl"
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
    "-maxLen",
    type=int,
    help="Maximum length of input tokens (tokenizer)",
    default= None
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size for dataloader",
    default=8
)

parser.add_argument(
    "-pretrainedModel",
    choices=["bert-base-uncased", "bert-base-cased"],
    help="Pretrained BERT model to use",
    default="bert-base-cased"
)

parser.add_argument(
    "-epochs",
    type=int,
    help="No. of epochs to train for",
    default=50
)


args = parser.parse_args()

debug = args.debug
logFile = args.log
trainFile = args.train
validFile = args.valid
testFile = args.test
trainValTest = args.trainValTest
relationsFile = args.relations
histogram = args.histogram
maxLen = args.maxLen
batchSize = args.batchSize
pretrainedModel = args.pretrainedModel
epochs = args.epochs

if logFile:
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

windowSize = 3

checkFile(trainFile, ".pkl")
if not trainValTest:
    checkFile(validFile, ".pkl")
    checkFile(testFile, ".pkl")
checkFile(mid2nameFile, ".pkl")

with open(trainFile, "rb") as f:
    train = pickle.load(f)

if not trainValTest:
    with open(validFile, "rb") as f:
        valid = pickle.load(f)
    with open(testFile, "rb") as f:
        test = pickle.load(f)

df = pd.DataFrame(train)
#Bert model expects integer targets, not strings
le = preprocessing.LabelEncoder()
df['relation'] = le.fit_transform(df['relation'])

if not trainValTest:
    df_train = df
    df_valid = pd.DataFrame(valid)
    df_test = pd.DataFrame(test)

    df_valid['relation'] = le.transform(df_valid['relation'])
    df_test['relation'] = le.transform(df_test['relation'])

if maxLen == None:
    maxLen = min(256, max([len(s) for s in df.text.to_numpy()]))

classes = np.unique(df.relation.to_numpy())

if histogram:
    sns.histplot(df, x="relation")
    plt.show()

tokenizer = transformers.BertTokenizer.from_pretrained(pretrainedModel)

if trainValTest:
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
    df_valid, df_test = train_test_split(df_test, test_size=0.6, random_state=13)

trainDataLoader = createDataLoader(df_train, tokenizer, maxLen, maxLen, debug)
validDataLoader = createDataLoader(df_valid, tokenizer, maxLen, maxLen, debug)
testDataLoader = createDataLoader(df_test, tokenizer, maxLen, maxLen, debug)

model = RelationClassifier(len(classes), pretrainedModel)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, correct_bias=False)
totalSteps = len(trainDataLoader)*epochs
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

    trainAcc, trainLoss = model.train(
        trainDataLoader,
        lossFunction,
        optimizer, 
        device, 
        scheduler,
        len(df_train)
    )

    logging.info(f"\tTrain Accuracy: {trainAcc}, Train loss: {trainLoss}")

    validAcc, validLoss = model.evaluate(
        validDataLoader,
        lossFunction,
        device, 
        len(df_valid)
    )

    logging.info(f"\Validation Accuracy: {validAcc}, Validation loss: {validLoss}")

    testAcc, testLoss = model.evaluate(
        validDataLoader,
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
        torch.save(model, "model.pth")
        bestAcc = validAcc