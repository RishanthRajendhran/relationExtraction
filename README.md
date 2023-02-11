<h1>
    Relation Extraction using Distant Supervision 
</h1>
<h4>
    Rishanth Rajendhran
</h4>
<br/>

<h5>
    Datsets
</h5>
<p>
    <ul>
        <li>
            <a href="https://www.microsoft.com/en-us/download/details.aspx?id=52312">
                FB15K-237 Knowledge Base Completion Dataset
            </a>
        </li>
        <li>
            <a href="https://github.com/xiaoling/figer/issues/6">
                Mapping between Freebase MIDs (Machine identifiers) and Wikipedia titles
            </a>
        </li>
        <li>
            Wikipedia articles (Wikipedia Python Module / Hunngingface Wikipedia dump)
        </li>
    </ul>
</p>

<h5>
    At the outset
</h5>
<p>
    This repository contains the code for a individual research project done as a part of the course
    CS 6390 Information Extraction by Professor Ellen Riloff at The University of Utah. 
    <br/>
    This project started out as an effort to build a system to extract family tree information from short stories but ended up getting a broader perspective primarily due to the lack of high-quality human annotated dataset for family tree/relationships extraction.
</p>
<br/>

<h5>
    Description
</h5>
<p>
    Relation extraction is a task focussed on extracting information about semantic relationship(s) between entities mentioned in a piece of text. For example, “Ryan, director of the Iskar award winning film ‘The thorns in my roses’, was seen yesterday in Salt Lake City.” encodes the following relationship: “The film ‘The thorns in my roses’ was directed by Ryan” or more succinctly (The thorns in my roses’ was directed by Ryan, directedBy, Ryan). In this project, we shall focus on extracting binary relationship between entities.
</p>
<p>
    Most natural language processing applications today make use of machine learning in some form. The performance of supervised machine learning systems, the most commonly used ML paradigm, depends largely on the quality and size of the labelled dataset. The lack of large human-annotated datasets for the task of relation extraction is an impediment. Other alternatives such as unsupervised ML and semi-supervised ML have their own limitations: The clusters formed using unsupervised techniques cannot directly mapped to real-world relations encoded in ontologies and databases; the seeds used in semi-supervised techniques could suffer from semantic drift. 
</p>
<p>
    To this end, a new ML paradigm called distant supervision was proposed by Mintz et al. (<a href="https://web.stanford.edu/~jurafsky/mintz.pdf">Link</a>) for the task of relation extraction. In the original paper, distant supervision was provided by relations and relation instances extracted from Freebase, a large knowledge base. of structured data. Once relation instances have been extracted from Freebase, sentences from Wikipedia articles containing both the entities involved in a relation instances are collected to create positive training instances. The intuition is that if two entities are related, then a sentence mentioning both these entities would possibly also mention the relation between these entities. To create negative instances, we take pairs of unrelated entities (entities that do not appear together in any relation instance) and extract sentences which contain both of these entities. 
</p>
<p>
    We can then generate feature vector representations for these instances using features such as named entity tags for the entities under consideration, context words and their part-of-speech tags etc. These feature vectors could then be used to train a classifier. Several improvements have been introduced in distant supervision over the years and we shall explore some of these methods in this project.
</p>
<br/>

<h5>
    Data Description
</h5>
<p>
    While the Freebase API has since been deprecated making it impossible to query it now for the data we need, the FB15k dataset, introduced by Bordes et al. (<a href="https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf">Link</a>), is available. The FB15K-237 Knowledge Base Completion Dataset which we shall be using for this project consists of a subset of relation instances from FB15K dataset which itself was derived from Freebase. This dataset consists of train, validation and test files. Each file consists of a set of relation instances of a large number of Freebase relations in the following format: “(entity_1) (relation) (entity_2)” in every line. 
</p>
<p>
    Given that the Freebase API is no longer available, we shall make use of a user-generated snapshot of the mapping between Freebase machine identifiers and wikipedia titles in-order to be able to extract relevant sentences from wikipedia articles. In this file, every line defines a mapping between an MID and wikipedia title in the following format: “(mid) (wikipediaTitle)”. Some MID’s can be associated with several Wikipedia titles to take all variants of spelling and synonyms into consideration. 
</p>
<p>
    We shall use the wikipedia python module to extract wikipedia articles for the various entities obtained from the above two datasets. The articles can then be split into sentences and the sentences can then be used for our application as outlined in the task description.
</p>
<br/>

<h5>
    Data Statistics
</h5>
<p>
    <table>
        <caption>
            FB15K-237 Knowledge Base Completion Dataset
        </caption>
        <tr>
            <th>
                Dataset
            </th>
            <th>
                #relationInstances
            </th>
            <th>
                #entities
            </th>
            <th>
                #relations
            </th>
            <th>
                Avg. no. of instances per entity
            </th>
            <th>
                % of repetitions
            </th>
        </tr>
        <tr>
            <th>
                Train
            </th>
            <td>
                272115
            </td>
            <td>
                14505
            </td>
            <td>
                237
            </td>
            <td>
                6277
            </td>
            <td>
                6.24
            </td>
        </tr>
        <tr>
            <th>
                Validation
            </th>
            <td>
                17535
            </td>
            <td>
                9809
            </td>
            <td>
                223
            </td>
            <td>
                715
            </td>
            <td>
                0.56
            </td>
        </tr>
        <tr>
            <th>
                Test
            </th>
            <td>
                20466
            </td>
            <td>
                10348
            </td>
            <td>
                224
            </td>
            <td>
                606
            </td>
            <td>
                0.72
            </td>
        </tr>
        <footer>
            If both (entity_1, relation, entity_2) and (entity_2, relation, entity_1) are present, it is considered a repetition.
        </footer>
    </table>
</p>
<p>
    <b>
        Mapping between Freebase MIDs (Machine identifiers) and Wikipedia titles
    </b>
    <br/>
    No. of mappings: 7606464
    <br/>
    No. of MIDs: 3305955
    <br/>
    Average no. of names per MID: 2.3
    <br/>
</p>
<br/>

<h5>
    Evaluation Plan
</h5>
<p>
    Standard evaluation metrics such as precision and recall are well suited for this task. They can be computed on a per-relation basis and then macro/micro-averaged for getting an overall number. 
</p>
<p>
    We shall make use of the test split of the FB15K-237 Knowledge Base Completion Dataset for the final evaluation. Much akin to the training process, we shall extract sentences mentioning both the entities involved in a test relation and then feed it to the trained classifier to obtain the relation label. We can then compare this label with the gold label present in the test file.
</p>
<br/>

<h5>
    Demo Plan
</h5>
<p>
    The input to the model shall be a pair of entities and a set of sentences containing both the entities. The output of the model would be the relation (as encoded in Freebase) between the two entities are reflected by the input sentences.
</p>
<p> 
    While the ultimate goal would be to build a user-friendly website (possibly using React and Flask) where the user can submit the input through a form and the response gets returned and presented in a readable manner, initial iterations would simply make use of the command-line terminal.
</p>
<br/>
<h5>
    External Tools
</h5>
<p>
    The major external tools/packages/modules used in this project:
    <ul>
        <li>
            wikipedia python API (for extracting wikipedia articles and summaries)
        </li>
        <li>
            PyTerrier (for inverted index)
        </li>
        <li>
            neuralcoref (for coreference resolution) (A neural network model using reinforcement learning based rewards based on <a href="https://aclanthology.org/D16-1245.pdf"> Deep Reinforcement Learning for Mention-Ranking Coreference Models by Clark et al.</a>)
        </li>
        <li>
            SpaCy (for named-entity recognition and dependency parsing)
        </li>
        <li>
            nltk (for word tokenization, sentence tokenization and part-of-speech tagging)
        </li>
        <li>
            huggingface (for transformers (BERT model))
        </li>
        <li>
            PyTorch (for linear layers, optimizers and loss functions)
        </li>
    </ul>
</p>
<h5>
    Pipeline
</h5>
<p>
    <ol>
        <li>
            Extract entities, relations and relation instances 
        </li>
        <li>
            Extract wikipedia articles for entities using information in mid2name mapping
        </li>
        <li>
            Apply coreference resolution over the extracted wikipedia articles
        </li>
        <li>
            Extract out sentences from coreference resolved wikipedia articles
        </li>
        <li>
            Build a inverted index treating sentences as documents 
        </li>
        <li>
            Generate positive examples by searching the inverted index for sentences containing both entities involved in a relation
        </li>
        <li>
            Generate negative examples from entity pairs not involved in a relation using a similar approach as in (6)
        </li>
        <li>
            Sample examples from the generated set of examples extracting out names for entities from sentences 
        </li>
        <li>
            Limit the number of examples per relation to ensure resulting set of examples are not too skewed
        </li>
        <li>
            Build a model using the examples
        </li>
    </ol> 
</p>
<h5>
    Files
</h5>
<p>
    <ul>
        <li>
            <h6>
                generateData.py
            </h6>
            <p>
                This file is used for the following operations:
                <ol>
                    <li>
                        Extract relations, entities and mid2name mappings from txt files
                    </li>
                    <li>
                        Pick relation instances for given set of relations/randomly sampled relations
                    </li>
                    <li>
                        Extract random wikipedia articles
                    </li>
                    <li>
                        Extract wikipedia articles/summaries for a given set of entities
                    </li>
                </ol>
            </p>
            <p>
                <h6>
                    Usage
                </h6>
                <pre>
usage: generateData.py [-h] [-debug] [-log LOG] [-map MAP] [-train TRAIN] [-valid VALID] [-test TEST] -mode {train,test,valid}
    [-load] [-pickRelations PICKRELATIONS] [-wiki] [-numSamples NUMSAMPLES] [-mid2name MID2NAME]
    [-entities ENTITIES] [-relations RELATIONS] [-wikiArticles WIKIARTICLES] [-article]
    [-maxInstsPerRel MAXINSTSPERREL] [-random]

options:
-h, --help            show this help message and exit
-debug                Boolean flag to enable debug mode
-log LOG              Path to file to print logging information
-map MAP              Path to TSV file containing mappings between MIDs and wikipedia titles
-train TRAIN          Path to txt train file containing relation instances
-valid VALID          Path to txt train file containing relation instances
-test TEST            Path to txt train file containing relation instances
-mode {train,test,valid}
                        Used to indicate the type of file being worked on (mappings would be extracted from mid2name only in train mode)
-load                 Boolean flag to indicate that mappings and relation/entities can be loaded
-pickRelations PICKRELATIONS
                        Flag to enable pickRelations mode and specify no. of relations to pick/text file containing relations to pick one per line
-wiki                 Boolean flag to enable wiki mode
-numSamples NUMSAMPLES
                        No. of relations sampled (Used for file naming purposes)
-mid2name MID2NAME    Path to file containing mid2name dictionary
-entities ENTITIES    Path to file containing entities dictionary
-relations RELATIONS  Path to file containing relations dictionary
-wikiArticles WIKIARTICLES
                        Path to file containing wiki articles list
-article              Boolean flag to be used in wiki mode to generate articles instead of summaries
-maxInstsPerRel MAXINSTSPERREL
                        Max. no. of instances per relation in pickRelation mode
-random               Boolean flag to be used in wiki mode to generate random articles
                </pre>
            </p>
        </li>
        <li>
            <h6>
                buildInvertedIndex.py
            </h6>
            <p>
                This file is used to perform the following operarions:
                <ol>    
                    <li>
                        Perform coreference resolution on wiki articles/summaries
                    </li>
                    <li>
                        Extract sentences from articles
                    </li>
                    <li>
                        Build a terrier index over all the articles
                    </li>
                </ol>
            </p>
            <p>
                <h6>
                    Usage
                </h6>
                <pre>
usage: generateData.py [-h] [-debug] [-log LOG] [-map MAP] [-train TRAIN] [-valid VALID] [-test TEST] -mode {train,test,valid}
                       [-load] [-pickRelations PICKRELATIONS] [-wiki] [-numSamples NUMSAMPLES] [-mid2name MID2NAME]
                       [-entities ENTITIES] [-relations RELATIONS] [-wikiArticles WIKIARTICLES] [-article]
                       [-maxInstsPerRel MAXINSTSPERREL] [-random]

options:
  -h, --help            show this help message and exit
  -debug                Boolean flag to enable debug mode
  -log LOG              Path to file to print logging information
  -map MAP              Path to TSV file containing mappings between MIDs and wikipedia titles
  -train TRAIN          Path to txt train file containing relation instances
  -valid VALID          Path to txt train file containing relation instances
  -test TEST            Path to txt train file containing relation instances
  -mode {train,test,valid}
                        Used to indicate the type of file being worked on (mappings would be extracted from mid2name only in train
                        mode)
  -load                 Boolean flag to indicate that mappings and relation/entities can be loaded
  -pickRelations PICKRELATIONS
                        Flag to enable pickRelations mode and specify no. of relations to pick/text file containing relations to
                        pick one per line
  -wiki                 Boolean flag to enable wiki mode
  -numSamples NUMSAMPLES
                        No. of relations sampled (Used for file naming purposes)
  -mid2name MID2NAME    Path to file containing mid2name dictionary
  -entities ENTITIES    Path to file containing entities dictionary
  -relations RELATIONS  Path to file containing relations dictionary
  -wikiArticles WIKIARTICLES
                        Path to file containing wiki articles list
  -article              Boolean flag to be used in wiki mode to generate articles instead of summaries
  -maxInstsPerRel MAXINSTSPERREL
                        Max. no. of instances per relation in pickRelation mode
  -random               Boolean flag to be used in wiki mode to generate random articles
                </pre>
            </p>
        </li>
        <li>
            <h6>
                generateExamples.py
            </h6>
            <p>
                This file is used to perform the following operarions:
                <ol>    
                    <li>
                        Generate examples by finding sentences containing both enities involved in a relation
                    </li>
                    <li>
                        Generate negative examples by finding sentences containing both enities not involved in any relation
                    </li>
                </ol>
            </p>
            <p>
                <h6>
                    Usage
                </h6>
                <pre>
usage: generateExamples.py [-h] [-debug] [-log LOG] [-mid2name MID2NAME] [-entities ENTITIES] [-relations RELATIONS] -invertedIndex INVERTEDINDEX -docs DOCS [-negative]
                           [-numExamples NUMEXAMPLES] [-out OUT] [-allEntities ALLENTITIES [ALLENTITIES ...]]

options:
  -h, --help            show this help message and exit
  -debug                Boolean flag to enable debug mode
  -log LOG              Path to file to print logging information
  -mid2name MID2NAME    Path to file containing mid2name dictionary
  -entities ENTITIES    Path to file containing entities dictionary
  -relations RELATIONS  Path to file containing relations dictionary
  -invertedIndex INVERTEDINDEX
                        Path to Terrier index folder
  -docs DOCS            Path to documents file (extension='.pkl')
  -negative             Boolean flag to generate negative examples
  -numExamples NUMEXAMPLES
                        No. of negative examples to generate
  -out OUT              Path to store examples (extension=.pkl)
  -allEntities ALLENTITIES [ALLENTITIES ...]
                        Negative examples: List of path to all entity files containing entities dictionary
                </pre>
            </p>
        </li>
        <li>
            <h6>
                sampleExamples.py
            </h6>
            <p>
                This file is used to perform the following operarions:
                <ol>    
                    <li>
                        Extract examples and transform them to a format than can be fed to a model
                    </li>
                </ol>
            </p>
            <p>
                <h6>
                    Usage
                </h6>
                <pre>
usage: sampleExamples.py [-h] [-debug] [-log LOG] [-mid2name MID2NAME] -examples EXAMPLES [EXAMPLES ...] -numSamplesPerReln NUMSAMPLESPERRELN [-out OUT]

options:
  -h, --help            show this help message and exit
  -debug                Boolean flag to enable debug mode
  -log LOG              Path to file to print logging information
  -mid2name MID2NAME    Path to file containing mid2name dictionary
  -examples EXAMPLES [EXAMPLES ...]
                        List of paths to files containing examples
  -numSamplesPerReln NUMSAMPLESPERRELN
                        No. of samples to pick per relation
  -out OUT              Path to sampled examples (extension=.pkl)
(relExtEnv) Rishanths-MacBook-Pro:Modules rishanthrajendhran$ 
                </pre>
            </p>
        </li>
        <li>
            <h6>
                buildModel.py
            </h6>
            <p>
                This file is used to perform the following operarions:
                <ol>    
                    <li>
                        Build the model
                    </li>
                    <li>
                        Train the model
                    </li>
                    <li>
                        Evaluate the model
                    </li>
                </ol>
            </p>
            <p>
                <h6>
                    Usage
                </h6>
                <pre>
usage: buildModel.py [-h] [-debug] [-log LOG] [-train TRAIN] [-valid VALID] [-test TEST] [-trainValTest] [-histogram] [-maxLen MAXLEN]
                     -batchSize BATCHSIZE -learningRate LEARNINGRATE [-pretrainedModel {bert-base-uncased,bert-base-cased}] [-epochs EPOCHS]
                     [-load LOAD] [-balance]

options:
  -h, --help            show this help message and exit
  -debug                Boolean flag to enable debug mode
  -log LOG              Path to file to print logging information
  -train TRAIN          Path to file containing training examples (extension=.pkl)
  -valid VALID          Path to file containing validation examples (extension=.pkl)
  -test TEST            Path to file containing test examples (extension=.pkl)
  -trainValTest         Boolean flag to split train set into train, validation and test set
  -histogram            Boolean flag to show histogram of examples
  -maxLen MAXLEN        Maximum length of input tokens (tokenizer)
  -batchSize BATCHSIZE  Batch size for dataloader
  -learningRate LEARNINGRATE
                        Learning rate for training
  -pretrainedModel {bert-base-uncased,bert-base-cased}
                        Pretrained BERT model to use
  -epochs EPOCHS        No. of epochs to train for
  -load LOAD            Path to file containing model to load
  -balance              Boolean flag to balance train dataset if not already balanced
                </pre>
            </p>
        </li>
        <li>
            <h6>
                testModel.py
            </h6>
            <p>
                This file is used to perform the following operarions:
                <ol>    
                    <li>
                        Test the model on a test file
                    </li>
                    <li>
                        Do a live demo
                    </li>
                    <li>
                        Evaluate baseline models
                    </li>
                </ol>
            </p>
            <p>
                <h6>
                    Usage
                </h6>
                <pre>
usage: testModel.py [-h] [-debug] [-log LOG] [-model MODEL] [-test TEST] [-maxLen MAXLEN] [-batchSize BATCHSIZE]
                    [-pretrainedModel {bert-base-uncased,bert-base-cased}] [-numClasses NUMCLASSES] [-live] [-le LABELENCODER]
                    [-example EXAMPLE] [-baseline {majority,random}] [-plotConfusion]

options:
  -h, --help            show this help message and exit
  -debug                Boolean flag to enable debug mode
  -log LOG              Path to file to print logging information
  -model MODEL          Path to file containing RelClassifier model (extension= .pt)
  -test TEST            Path to file containing test examples (extension=.pkl)
  -maxLen MAXLEN        Maximum length of input tokens (tokenizer)
  -batchSize BATCHSIZE  Batch size for dataloader
  -pretrainedModel {bert-base-uncased,bert-base-cased}
                        Pretrained BERT model to use
  -numClasses NUMCLASSES
                        No. of classes the input model is built for
  -live                 Boolean flag to enable live demo mode
  -le LABELENCODER, --labelEncoder LABELENCODER
                        Path to file containing label encoder object
  -example EXAMPLE      Example sentence to test the model on in live demo mode
  -baseline {majority,random}
                        Test Baseline models
  -plotConfusion        Boolean flag to plot confusion matrix after evaluation
                </pre>
            </p>
        </li>
    </ul>
</p>

<h5>
    Literature Review
</h5>
<p>
    <b>
        Distant Supervision
    </b>
    <ol>
        <li>
            <a href="https://dl.acm.org/doi/10.5555/1690219.16902872">
                Distant supervision for relation extraction without labeled data
            </a>
        </li>
        <li>
            <a href="https://aclanthology.org/D12-1042/">
                Multi-instance Multi-label Learning for Relation Extraction
            </a>
        </li>
        <li>
            <a href="https://www.semantic-web-journal.net/content/relation-extraction-web-using-distant-supervision">
                Relation Extraction from the Web using Distant Supervision
            </a>
        </li>
        <li>
            <a href="https://aclanthology.org/C16-1139/">
                Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks
            </a>
        </li>
        <li>
            <a href="https://dl.acm.org/doi/10.5555/3298483.3298679">
                Distant supervision for relation extraction with sentence-level attention and entity descriptions
            </a>
        </li>
        <li>
            <a href="https://dl.acm.org/doi/10.1609/aaai.v33i01.33017418">
                Distant supervision for relation extraction with linear attenuation simulation and non-IID relevance embedding
            </a>
        </li>
    </ol>
    <b>
        Other
    </b>
    <ol>
        <li>
            <a href="https://www.researchgate.net/publication/265006408_A_Review_of_Relation_Extraction">
                A Review of Relation Extraction
            </a>
        </li>
        <li>
            <a href="https://aclanthology.org/2020.aacl-main.75/">
                More Data, More Relations, More Context and More Openness: A Review and Outlook for Relation Extraction
            </a>
        </li>
    </ol>
</p>