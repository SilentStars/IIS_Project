from gestureClassification import _concat_data,_get_Xs_ys,_get_label_encoder,_encode_labels,_build_model,_train_model


if __name__ == '__main__':
    data = _concat_data()
    X_train,X_val,X_test,y_train,y_val,y_test = _get_Xs_ys(data)

    encoder = _get_label_encoder(y_train)
    y_train_encoded = _encode_labels(y_train,encoder)
    y_val_encoded = _encode_labels(y_val,encoder)
    y_test_encoded = _encode_labels(y_test,encoder)

    model = _build_model()
    history = _train_model(model,X_train,y_train_encoded,X_val,y_val_encoded,X_test,y_test_encoded)

    model.save('./models/mlp')