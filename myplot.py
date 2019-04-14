def plt_init(plt):
    plt.rcParams['image.interpolation'] = 'bilinear'
    plt.rcParams['figure.dpi'] = 300


def draw_image_array(row, col, imgs, spacing, start=0):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    shape = int(imgs[0].shape[0]**0.5)
    shape = (shape, shape)
    _imgs = np.array(
        list(map(lambda x: np.reshape(x, shape), imgs)),
        dtype=np.uint8
    )
    h_blank = np.zeros((shape[1], spacing), dtype=np.uint8)
    h_blank.fill(255)
    v_blank = np.zeros((spacing, shape[0] * col + spacing * (col - 1)), dtype=np.uint8)
    v_blank.fill(255)
    canvas = None
    for i in range(row):
        rowdata = _imgs[i * col + start]
        for j in range(col - 1):
            rowdata = np.concatenate((rowdata, h_blank, _imgs[j + i * col + 1 + start]), axis=1)
        if i == 0:
            canvas = rowdata
        else:
            canvas = np.concatenate((canvas, v_blank, rowdata), axis=0)
    plt.imshow(canvas, cmap='gray')
    plt.xticks([])
    plt.yticks([])


def extract_specific_categories(path, column_name, labels):
    '''
    path = csv file path
    column_name = label name with respect to pandas' column name
    labels = list of which categories you want to extract from dataset
    '''
    import pandas as pd
    import numpy as np

    data = pd.read_csv(path)
    extracted_categorys = []
    extracted_categorys_labels = []
    for label in labels:
        category = data.loc[data[column_name] == label]
        category_label = category[column_name].values
        category = category.drop(column_name, 1).values
        # print(category.shape)
        if len(extracted_categorys) == 0:
            extracted_categorys = category
            extracted_categorys_labels = category_label
        else:
            extracted_categorys = np.concatenate((extracted_categorys, category), axis=0)
            extracted_categorys_labels = np.concatenate((extracted_categorys_labels, category_label), axis=0)

    return extracted_categorys, extracted_categorys_labels


def extract_specific_amount(data_x, data_y, labels, amount):
    import numpy as np
    def _extract(x): return data_x[data_y == x][0:amount]
    def _extract_labels(x): return data_y[data_y == x][0:amount]
    return np.concatenate(tuple(map(_extract, labels))), np.concatenate(tuple(map(_extract_labels, labels)))


def plot_decision_boundary(pred_func, X):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z = (Z + 0.5) / 10
    # cmap = cm.get_cmap('tab10', 10)

    plt.contourf(xx, yy, Z, levels=5, cmap=plt.cm.Spectral)
    contour_cb = plt.colorbar()
    contour_cb.set_label('decision plane')
    contour_cb.set_ticklabels([0, 1, 3, 6, 8, 9], update_ticks=True)
    plt.contour(xx, yy, Z, colors='black', linewidths=0.2)
    # contour_cb.set_ticks([])
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)


def plot_multiclasses_2d_distribution(data, label, pick, marker, min_dist=0.08, subplot=None):
    import matplotlib.pyplot as plt
    import numpy as np
    _data = []
    for x in pick:
        _data.append(data[label == x])

    _data = np.array(_data)
    print(_data.shape)
    _mask = np.ones(_data.shape[0:2], dtype=bool)
    _len = [x.shape[0] for x in _data]
    # 去除距离过近的点
    for i in range(_data.shape[0]):
        for j in range(_data.shape[1] - 1):
            dist = _data[i] - _data[i][j]
            dist[j] = [1, 1]
            dist = dist[_mask[i]]
            dist = abs(dist[:, 0]) + abs(dist[:, 1])
            # print(dist.min())
            if dist.min() < min_dist:
                _mask[i][j] = False
                _len[i] = _len[i] - 1

    _data = _data[_mask]
    print('_data shape:', _data.shape)
    for i in range(1, len(_len)):
        _len[i] = _len[i] + _len[i - 1]
    _len = np.concatenate(([0], _len), axis=0)

    __data = []
    for i in range(len(_len) - 1):
        __data.append(_data[_len[i]:_len[i + 1]])
    _data = __data

    for i, x in enumerate(_data):
        plt.scatter(x[:, 0], x[:, 1], marker=marker[i], c='k', s=5, linewidths=0.5, alpha=0.5, label=str(pick[i]))
    # plt.yticks([])
    # plt.xticks([])
    plt.legend(loc="upper left", fontsize=6, markerscale=1.5)
