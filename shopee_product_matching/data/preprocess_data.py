import pandas as pd
from data.datamodule import ShopeeDataModule


def create_datamodule(
    train_df, val_df, image_dir: str, image_size, num_workers=4, prefetch_factor=2, P=16, K=4
) -> ShopeeDataModule:
    """
    Функция для создания DataModule

    Returns:
        ShopeeDataModule: настроенный DataModule
    """

    datamodule = ShopeeDataModule(
        train_df=train_df,
        val_df=val_df,
        image_dir=image_dir,
        image_size=image_size,
        train_scale=1.1,  # случайное масштабирование для тренировки
        val_scale=1.0,
        P=16,
        K=4,
        num_workers=num_workers,  # 4
        prefetch_factor=prefetch_factor,  # 2
        pin_memory=True,
    )

    return datamodule


def preprocess_data_for_model(
    df_path,
    little_filter_count,
    tokenizer,
    train_ratio,
    image_dir,
    num_workers,
    prefetch_factor,
    p_sampler,
    k_sampler,
):
    df = pd.read_csv(df_path)

    if little_filter_count:
        count_users = df.label_group.value_counts().reset_index()
        very_little_product = count_users[
            count_users["count"] <= little_filter_count
        ].label_group.unique()
        df = df[~df["label_group"].isin(very_little_product)]

    texts = list(df["title"].apply(lambda o: str(o)).values)
    text_encodings = tokenizer(texts, padding=True, truncation=True, max_length=64)

    df["input_ids"] = text_encodings["input_ids"]
    df["attention_mask"] = text_encodings["attention_mask"]

    train_size = int(len(df.label_group.unique()) * train_ratio)

    train_labels = df.label_group.unique()[:train_size]
    val_labels = df.label_group.unique()[train_size:]

    df_train = df[df["label_group"].isin(train_labels)]
    df_test = df[df["label_group"].isin(val_labels)]

    return create_datamodule(
        train_df=df_train,
        val_df=df_test,
        image_dir=image_dir,
        image_size=420,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        P=p_sampler,
        K=k_sampler,
    )
