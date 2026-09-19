import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "Simple_Recommendation_System/Clean_Movielens.csv"
CACHE_DIR = "Simple_Recommendation_System/cache"
SEED = 42
MIN_USER_RATINGS = 20
N_ACTIVE_USERS = 300
TEST_FRACTION = 0.2
TOP_MOVIES_CONTENT = 5000
TOP_MOVIES_COLLAB = 2000
K = 5

ratings = pd.read_csv(DATA_PATH)
title_of = ratings.drop_duplicates("movieId").set_index("movieId")["title"].to_dict()


def load_or_build(name, builder):
    path = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            print(f"Loaded {name} from cache")
            return pickle.load(f)
    print(f"Building {name} ...")
    obj = builder()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def clear_cache():
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))
        print("Cache cleared")


def build_content_model(frame):
    popular = frame.groupby("movieId")["userId"].count().nlargest(TOP_MOVIES_CONTENT).index
    movies = frame[frame["movieId"].isin(popular)][["movieId", "title", "genres"]]
    movies = movies.drop_duplicates("title").reset_index(drop=True)
    vectors = TfidfVectorizer().fit_transform(movies["genres"])
    return movies, cosine_similarity(vectors)


def recommend_movies(movie_title, movies, similarity):
    if movie_title not in movies["title"].values:
        return "Movie not found!"
    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [movies["title"][i] for i, _ in scores[1 : K + 1]]


def build_collab_model(frame):
    active = frame["userId"].value_counts()
    active = active[active >= MIN_USER_RATINGS].index
    sub = frame[frame["userId"].isin(active)]
    popular = sub.groupby("movieId")["userId"].count().nlargest(TOP_MOVIES_COLLAB).index
    sub = sub[sub["movieId"].isin(popular)]
    matrix = sub.pivot_table(index="userId", columns="movieId", values="ratings").fillna(0)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    return sim_df, matrix


def predict_matrix(sim, matrix):
    predicted = sim @ matrix
    counts = sim @ (matrix > 0)
    predicted = np.divide(predicted, counts, out=np.zeros_like(predicted), where=counts > 0)
    predicted[~np.isfinite(predicted)] = 0
    return predicted


def recommend_for_user(user_id, sim_df, matrix, top_k=K):
    if user_id not in matrix.index:
        return "User not found! (needs >= " + str(MIN_USER_RATINGS) + " ratings to be modeled)"
    pred = predict_matrix(sim_df.values, matrix.values)
    user_idx = matrix.index.get_loc(user_id)
    scores = pred[user_idx].copy()
    scores[matrix.values[user_idx] > 0] = -np.inf
    top = np.argsort(scores)[::-1][:top_k]
    return [title_of[matrix.columns[i]] for i in top if scores[i] > -np.inf]


def evaluate():
    top_users = ratings["userId"].value_counts().head(N_ACTIVE_USERS).index
    train_parts, test_parts = [], []
    for uid in top_users:
        rows = ratings[ratings["userId"] == uid].sample(frac=1, random_state=SEED)
        n_test = max(1, int(len(rows) * TEST_FRACTION))
        test_parts.append(rows.iloc[:n_test])
        train_parts.append(rows.iloc[n_test:])
    train = pd.concat(train_parts)
    test = pd.concat(test_parts)

    active = train["userId"].value_counts()
    active = active[active >= 5].index
    train = train[train["userId"].isin(active)]
    test = test[test["userId"].isin(active)]

    matrix = train.pivot_table(index="userId", columns="movieId", values="ratings").fillna(0)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0)
    pred, counts = sim @ matrix.values, sim @ (matrix.values > 0)
    pred = np.divide(pred, counts, out=np.zeros_like(pred), where=counts > 0)
    pred[~np.isfinite(pred)] = 0

    rows = test[test["movieId"].isin(matrix.columns)]
    iu = rows["userId"].map(matrix.index.get_loc).values
    im = rows["movieId"].map(matrix.columns.get_loc).values
    y_pred, y_true = pred[iu, im], rows["ratings"].values
    valid = (counts[iu, im] > 0) & np.isfinite(y_pred)
    rmse = float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))
    mae = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
    baseline = float(np.sqrt(np.mean((y_true[valid] - train["ratings"].mean()) ** 2)))

    unseen = matrix.values == 0
    scores = pred.copy()
    scores[~unseen] = -np.inf
    top_idx = np.argsort(scores, axis=1)[:, -K:][:, ::-1]
    test_items = {g: set(v.tolist()) for g, v in test.groupby("userId")["movieId"] if g in matrix.index}
    precisions, hit = [], 0
    for i, uid in enumerate(matrix.index):
        wanted = test_items.get(uid, set()) - set(matrix.columns[matrix.values[i] > 0])
        if not wanted:
            continue
        recs = [matrix.columns[j] for j in top_idx[i] if np.isfinite(pred[i, j])]
        hits = [m for m in recs if m in wanted]
        precisions.append(len(hits) / K)
        hit += len(hits) > 0

    print(f"\n[user-based CF evaluated on {N_ACTIVE_USERS} most active users, per-user {1 - TEST_FRACTION:.0%}/{TEST_FRACTION:.0%} split]")
    print(f"model: {matrix.shape[0]} users x {matrix.shape[1]} movies")
    print(f"test pairs: {len(rows)}   evaluated: {valid.sum()}   RMSE: {rmse:.3f}   MAE: {mae:.3f}")
    print(f"baseline (always predict mean={train['ratings'].mean():.2f}) RMSE: {baseline:.3f}")
    if precisions:
        print(f"precision@{K}: {np.mean(precisions):.3f}   hit-rate: {hit / len(precisions):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Movie recommender (content + user-based collaborative)")
    parser.add_argument("--recommend", metavar="TITLE", help="content-based: similar movies to TITLE")
    parser.add_argument("--for-user", metavar="USER_ID", type=int, help="collaborative: recommend for USER_ID")
    parser.add_argument("--evaluate", action="store_true", help="train/test RMSE, MAE, precision@K")
    parser.add_argument("--rebuild-cache", action="store_true", help="clear cached similarity models")
    args = parser.parse_args()

    if args.rebuild_cache:
        clear_cache()

    movies, similarity = load_or_build("content", lambda: build_content_model(ratings))
    sim_df, matrix = load_or_build("collab", lambda: build_collab_model(ratings))

    if not (args.recommend or args.for_user or args.evaluate):
        args.recommend, args.for_user, args.evaluate = "Sense and Sensibility (1995)", 175325, True

    if args.recommend:
        print("Content-Based for", repr(args.recommend), ":", recommend_movies(args.recommend, movies, similarity))
    if args.for_user:
        print("User-Based for user", args.for_user, ":", recommend_for_user(args.for_user, sim_df, matrix))
    if args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()