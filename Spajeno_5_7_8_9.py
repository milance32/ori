# ------------------- 5 -------------------
import numpy as np

class NB_Sentiment_Classifier:
    def __init__(self, texts:list, sentiments:list) -> None:
        assert len(texts) == len(sentiments), 'Za svaki tekst postoji tačno jedan sentiment.'
        self.texts = texts
        self.sentiments = sentiments
        self.pos_word_counts = {} 
        self.neg_word_counts = {}
        self.text_counts = {'pos': 0, 'neg': 0} # broj ✅ tekstova i broj ❌ tekstova
        self.n_words = {'pos': 0, 'neg': 0} # ukupan broj ✅ reči i ukupan broj ❌ reči
        self.prior = {'pos': 0, 'neg': 0} # P(✅) i P(❌)

    def _preprocess(self, text:str) -> str:
        '''Preprocess and returns text.'''
        import re
        text = re.sub(r'[^\w\s]', '', text) # uklonimo znakove
        words = text.lower().split() # svedemo na mala slova i podelimo na reči
        return words
    
    def fit(self) -> None:
        '''Train a classifier.'''
        # pravimo tabelu ponavljanja za svaku rec - Bag-of-words
        for text, sentiment in zip(self.texts, self.sentiments):
            words = self._preprocess(text)
            for word in words: 
                if sentiment == 'pos': self.pos_word_counts[word] = self.pos_word_counts.get(word, 0) + 1
                if sentiment == 'neg': self.neg_word_counts[word] = self.neg_word_counts.get(word, 0) + 1

        # broj ✅ tekstova i broj ❌ tekstova
        self.text_counts['pos'] = len([s for s in self.sentiments if s=='pos'])
        self.text_counts['neg'] = len([s for s in self.sentiments if s=='neg'])

        # ukupan broj ✅ reči i ukupan broj ❌ reči
        self.n_words['pos'] = sum(self.pos_word_counts.values())
        self.n_words['neg'] = sum(self.neg_word_counts.values())
        
        # nadji P(✅) i P(❌)
        n_total_texts = sum(self.text_counts.values())
        self.prior['pos'] = self.text_counts['pos'] / n_total_texts
        self.prior['neg'] = self.text_counts['neg'] / n_total_texts


    def predict(self, text:str) -> tuple[float, float]:
        '''Returns a list of: [P(text|✅), P(❌|text)].'''
        words = self._preprocess(text)
        p_words_given_pos_sentiment = []
        p_words_given_neg_sentiment = []
        # iteraramo kroz sve reci u recenici i racunamo P(word|✅)) i P(word|❌)
        for word in words:
            # verovatnoća da se reč nađe u pozitivnoj recenziji
            p_word_given_pos = self.pos_word_counts.get(word, 0) + 1 / (self.n_words['pos'] + len(self.pos_word_counts)) # Laplace Smoothing
            p_words_given_pos_sentiment.append(p_word_given_pos)

            # verovatnoća da se reč nađe u negativnoj recenziji
            p_word_given_neg = self.neg_word_counts.get(word, 0) + 1 / (self.n_words['neg'] + len(self.neg_word_counts)) # Laplace Smoothing
            p_words_given_neg_sentiment.append(p_word_given_neg)

        # računamo P(text|✅) i P(text|❌) tako sto pomnozimo verovatnoću za svaku reč
        p_text_given_pos = np.prod(p_words_given_pos_sentiment)
        p_text_given_neg = np.prod(p_words_given_neg_sentiment)

        # iskoristimo Bajesovu formulu da nadjemo P(✅|text) i P(❌|text)
        p_text_is_pos = self.prior['pos'] * p_text_given_pos
        p_text_is_neg = self.prior['neg'] * p_text_given_neg
        
        return p_text_is_pos, p_text_is_neg


if __name__ == '__main__':
    reviews = {
    'The movie was great': 'pos',
    'That was the greatest movie!': 'pos',
    'I really enjoyed that great movie.': 'pos',
    'The acting was terrible': 'neg',
    'The movie was not great at all...': 'neg'}

    reviews_texts = list(reviews.keys())
    reviews_sentiments = list(reviews.values())
    clf = NB_Sentiment_Classifier(reviews_texts, reviews_sentiments)
    clf.fit()
    
    text = 'The movie was terrible, terrible, terrible,...'
    p_text_is_pos, p_text_is_neg = clf.predict(text)
    
    print(f'P(✅|{text}) = {p_text_is_pos:.5f}')
    print(f'P(❌|{text}) = {p_text_is_neg:.5f}')
    if p_text_is_pos > p_text_is_neg: print('Recenzija je pozitivna')
    else: print('Recenzija je negativna')

# ------------------- 7 -------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Učitavamo skup podataka
# Želimo da klasifikujemo cvetove na 
# 0. Iris Setosa
# 1. Iris Versicolour
# 2. Iris Virginica
# X obeležja predstavljaju:
# Sepal Length, Sepal Width, Petal Length and Petal Width

iris = load_iris()
X = iris.data
y = iris.target
# Delimo na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(set(y_train))
model = MLP(input_size, hidden_size, num_classes)

# Loss funkcija i optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening petlja
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluacija na testnom skupu
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on test set: {accuracy:.2f}')


# ------------------- 8 -------------------
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])

train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                             train = True,
                                             transform = all_transforms,
                                             download = True)


test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = all_transforms,
                                            download=True)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)



class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = ConvNeuralNet(num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))


# ------------------- 9 -------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Učitavamo skup podataka
# Želimo da klasifikujemo cvetove na 
# 0. Iris Setosa
# 1. Iris Versicolour
# 2. Iris Virginica
# X obeležja predstavljaju:
# Sepal Length, Sepal Width, Petal Length and Petal Width

iris = load_iris()
X = iris.data
y = iris.target
# Delimo na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(set(y_train))
model = MLP(input_size, hidden_size, num_classes)

# Loss funkcija i optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening petlja
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluacija na testnom skupu
with torch.no_grad():
    model.eval()
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on test set: {accuracy:.2f}')


