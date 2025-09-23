import fire


class App:
    def train(self, config="examples/", **kwargs):
        print("kwargs", kwargs, {k: type(v) for k, v in kwargs.items()})


if __name__ == "__main__":
    fire.Fire(App)
