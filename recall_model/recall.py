from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# 初始化 SparkSession
spark = SparkSession.builder.appName("TowTowerModel").getOrCreate()

# 加载数据集
train_df = spark.read.parquet("./train_df.parquet")
val_df = spark.read.parquet("./validation_data.parquet")
val_df = val_df.select("user_id", "movie_id", "gender", 'age_vector', 'occupation_vector', 'genres_vector')
val_df = val_df.withColumn("label", lit(1))

# 模型参数
user_num = train_df.select('user_id').distinct().count()
movie_num = train_df.select('movie_id').distinct().count()
age_categories = len(train_df.select("age_vector").first()[0])
occupation_categories = len(train_df.select("occupation_vector").first()[0])
genres_categories = len(train_df.select("genres_vector").first()[0])

# 将合并后的 DataFrame 转化为 RDD 或其他需要的格式以便进一步训练
train_df_rdd = train_df.rdd
val_df_rdd = val_df.rdd


class MovieLensDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        # 使用DataFrame.count()来获取数据集的大小
        return self.data_df.count()

    def __getitem__(self, idx):
        row = self.data_df.collect()[idx]
        user_id = torch.tensor(row['user_id'], dtype=torch.long).unsqueeze(0).unsqueeze(1)
        gender = torch.tensor(row['gender'], dtype=torch.long).unsqueeze(0).unsqueeze(1)
        # 假设age是one-hot编码，需要先转换为一维张量
        age_vector = torch.tensor(row['age_vector'], dtype=torch.long).unsqueeze(0)
        occupation_vector = torch.tensor(row['occupation_vector'], dtype=torch.long).unsqueeze(0)
        movie_id = torch.tensor(row['movie_id'], dtype=torch.long).unsqueeze(0).unsqueeze(1)
        genres_vector = torch.tensor(row['genres_vector'], dtype=torch.long).unsqueeze(0)
        label = torch.tensor(row['label'], dtype=torch.float32).unsqueeze(0)

        # 合并user_features  
        user_features = torch.cat([user_id, gender, age_vector, occupation_vector], dim=1)  
        # movie_features保持为两个独立的Tensor  
        movie_features = torch.cat([movie_id, genres_vector], dim=1)  

        return user_features, movie_features, label  


train_pdf = train_df.toPandas()
val_pdf = val_df.toPandas()

# 创建MovieLensDataset实例
train_dataset = MovieLensDataset(train_pdf)
val_dataset = MovieLensDataset(val_pdf)
# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# 创建FeatureEmbedder类
class FeatureEmbedder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(FeatureEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# 双塔模型
class TwoTowerModel(nn.Module):
    def __init__(self, user_num, movie_num, age_categories, occupation_categories, genres_categories, embed_dim, hidden_dim):
        super(TwoTowerModel, self).__init__()
        # 用户塔
        self.user_embed = FeatureEmbedder(user_num, embed_dim)
        self.user_fc = nn.Linear(embed_dim + 1 + age_categories + occupation_categories, hidden_dim)  # 全连接层
        
        # 物品塔
        self.movie_embed = FeatureEmbedder(movie_num, embed_dim)
        self.movie_fc = nn.Linear(embed_dim + genres_categories, hidden_dim)  # 全连接层

    def forward(self, user_features, movie_features):
        # 用户塔的嵌入
        user_embedded = torch.cat([
            self.user_embed(user_features[:, 0].long()),
            user_features[:, 1],
            user_features[:, 2:age_categories+1],
            user_features[:, age_categories+1:]
            ], dim=1)

        # 物品塔的嵌入
        movie_embedded = torch.cat([
            self.movie_embed(movie_features[:, 0].long()),
            movie_features[:, 1:]
            ], dim=1)

        return user_embedded, movie_embedded

# 余弦损失函数
def cosine_loss(user_embedded, movie_embedded, labels):
    return torch.nn.CosineEmbeddingLoss()(user_embedded, movie_embedded, labels)


# 假设age_categories是年龄独热编码的类别数
user_num = user_num
movie_num = movie_num
age_categories = age_categories
occupation_categories = occupation_categories
genres_categories = genres_categories
embed_dim = 128
hidden_dim = 64

# 初始化双塔模型
recall_model = TwoTowerModel(
    user_num, movie_num, age_categories, occupation_categories, genres_categories, embed_dim, hidden_dim)
optimizer = torch.optim.Adam(recall_model.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将模型移动到GPU上
recall_model.to(device)

num_epochs = 10

# 训练双塔模型
def train_model(model, train_loader, val_loader):
    model.train()
    for epoch in range(num_epochs):
        for user_features, movie_features, labels in train_loader:
            # 将数据移动到GPU
            user_features = user_features.to(device)
            movie_features = movie_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            user_embedded, movie_embedded = model(user_features, movie_features)
            loss = cosine_loss(user_embedded, movie_embedded, labels)
            loss.backward()
            optimizer.step()

        # 验证过程
        model.eval()
        with torch.no_grad():
            for user_features, movie_features, labels in val_loader:
                # 将数据移动到GPU
                user_features = user_features.to(device)
                movie_features = movie_features.to(device)
                labels = labels.to(device)

                user_embedded, movie_embedded = model(user_features, movie_features)
                val_loss = cosine_loss(user_embedded, movie_embedded, labels)
                # 打印验证损失
                print(f'Epoch {epoch}, Validation Loss: {val_loss.item()}')

# 训练模型
train_model(recall_model, train_loader, val_loader)