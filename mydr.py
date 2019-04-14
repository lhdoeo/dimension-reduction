def eval_dimension_reduction_method(method, n_components, data, label, params, kfold=0):
    import time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np
	# 对数据进行归一化
    normalizer = MinMaxScaler()
    data = normalizer.fit_transform(data)

    if kfold != 0:
        kf = StratifiedKFold(n_splits=kfold, random_state=0)
        final_score = []
        final_time = []
        for train_index, test_index in kf.split(data, label):
            train_data, test_data = data[train_index], data[test_index]
            train_label, test_label = label[train_index], label[test_index]

            start = time.time()

            if method == 'pca':
                pca = PCA(n_components=n_components, whiten=False, svd_solver='auto', random_state=0)
                reduced_train_data = pca.fit_transform(train_data)
            if method == 'iso':
                iso = Isomap(n_neighbors=params.n_neighbors, n_components=n_components, n_jobs=-1)
                reduced_train_data = iso.fit_transform(train_data)
            if method == 'lle':
                lle = LocallyLinearEmbedding(n_neighbors=params['n_neighbors'], n_components=n_components, method=params['method'], n_jobs=-1, random_state=0)
                reduced_train_data = lle.fit_transform(train_data)
            if method == 'mds':
                mds = MDS(n_components=n_components, n_init=1, random_state=0)
                reduced_train_data = mds.fit_transform(train_data)
            if method == 'le':
                le = SpectralEmbedding(n_components=n_components, random_state=0, n_jobs=-1)
                reduced_data = le.fit_transform(train_data)
            if method == 'tsne':
                tsne = TSNE(n_components=n_components, random_state=0)
                reduced_data = tsne.fit_transform(train_data)

            end = time.time()
			# 对降维数据进行标准化
            scaler = StandardScaler()
            reduced_train_data = scaler.fit_transform(reduced_train_data)

            svc = SVC(kernel='rbf', gamma='scale', random_state=0, decision_function_shape='ovo')
            svc.fit(reduced_train_data, train_label)
            score = svc.score(reduced_train_data, train_label)

            final_score.append(score)
            final_time.append(end - start)
            print('-', end='')
        final_score = np.mean(final_score)
        final_time = np.mean(final_time)
        print('{}+svm cost {:.3f} s score {}'.format(method, final_time, final_score))

    else:
        if method == 'pca':
            pca = PCA(n_components=n_components, whiten=False, svd_solver='auto', random_state=0)
            learn_start = time.time()
            pca.fit(data)
            learn_end = time.time()
            inference_start = time.time()
            reduced_data = pca.transform(data)
            inference_end = time.time()
        if method == 'iso':
            iso = Isomap(n_neighbors=params['n_neighbors'], n_components=n_components, n_jobs=-1)
            learn_start = time.time()
            iso.fit(data)
            learn_end = time.time()
            inference_start = time.time()
            reduced_data = iso.transform(data)
            inference_end = time.time()
        if method == 'lle':
            lle = LocallyLinearEmbedding(n_neighbors=params['n_neighbors'], n_components=n_components, method=params['method'], n_jobs=-1, random_state=0)
            learn_start = time.time()
            lle.fit(data)
            learn_end = time.time()
            inference_start = time.time()
            reduced_data = lle.transform(data)
            inference_end = time.time()
        if method == 'mds':
            mds = MDS(n_components=n_components, n_init=1, random_state=0)
            inference_start = time.time()
            reduced_data = mds.fit_transform(data)
            inference_end = time.time()
        if method == 'le':
            le = SpectralEmbedding(n_components=n_components, random_state=0, n_jobs=-1)
            inference_start = time.time()
            reduced_data = le.fit_transform(data)
            inference_end = time.time()
        if method == 'tsne':
            tsne = TSNE(n_components=n_components, random_state=0)
            inference_start = time.time()
            reduced_data = tsne.fit_transform(data)
            inference_end = time.time()

        scaler = StandardScaler()
        reduced_data = scaler.fit_transform(reduced_data)

        svc = SVC(kernel='rbf', gamma='scale', random_state=0, decision_function_shape='ovo')
        svc.fit(reduced_data, label)
        score = svc.score(reduced_data, label)

        if method == 'pca':
            print('learn time:{:.3f} inference time:{:.3f} score:{}'.format((learn_end - learn_start),(inference_end - inference_start),score))
            return normalizer, pca, scaler, svc, reduced_data, label
        if method == 'iso':
            print('learn time:{:.3f} inference time:{:.3f} score:{}'.format((learn_end - learn_start),(inference_end - inference_start),score))
            return normalizer, iso, scaler, svc, reduced_data, label
        if method == 'lle':
            print('learn time:{:.3f} inference time:{:.3f} score:{}'.format((learn_end - learn_start),(inference_end - inference_start),score))
            return normalizer, lle, scaler, svc, reduced_data, label
        if method == 'mds':
            print('inference time:{:.3f} score:{}'.format((inference_end - inference_start),score))
            return normalizer, mds, scaler, svc, reduced_data, label
        if method == 'le':
            print('inference time:{:.3f} score:{}'.format((inference_end - inference_start),score))
            return normalizer, le, scaler, svc, reduced_data, label
        if method == 'tsne':
            print('inference time:{:.3f} score:{}'.format((inference_end - inference_start),score))
            return normalizer, tsne, scaler, svc, reduced_data, label
