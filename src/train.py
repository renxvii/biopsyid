from models.unet import BiopsyUNet


def main():

    model = BiopsyUNet()

    print(model)

    print()

    print("Model successfully initialized.")


if __name__ == "__main__":

    main()