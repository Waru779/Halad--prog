import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import ttkbootstrap as ttk


class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lineáris regresszió Tkinterrel")
        self.root.minsize(450, 330)

        self.data = None
        self.model = None

        # ALAP KINÉZERT
        main = ttk.Frame(root, padding=20)
        main.grid(row=0, column=0, sticky="nsew")

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        #GOMBOK
        ttk.Button(main, text="Adatok betöltése (CSV)", command=self.load_csv).pack(fill="x", pady=5)
        ttk.Button(main, text="Új adatok generálása", command=self.generate_new_data).pack(fill="x", pady=5)
        ttk.Button(main, text="Modell tanítása", command=self.train_model).pack(fill="x", pady=5)
        ttk.Button(main, text="Grafikon megjelenítése", command=self.show_plot).pack(fill="x", pady=5)

        #Eredmény
        self.result_label = ttk.Label(main, text="Nincs adat", justify="center", font=("Arial", 11),anchor="center")
        self.result_label.pack(pady=15, fill="x")

        #Alsó státusz
        self.status = ttk.Label(root, text="Készen áll.", anchor="center", bootstyle="secondary" )
        self.status.grid(row=1, column=0, sticky="ew")

    #Adatok betöltése
    def load_csv(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV fájl", "*.csv"), ("Összes fájl", "*.*")]
        )
        if filepath:
            try:
                self.data = pd.read_csv(filepath)
                if self.data.shape[1] < 2:
                    raise ValueError("Legalább 2 oszlop kell (X és Y).")

                self.status.config(text="CSV betöltve!", bootstyle="info")
                self.result_label.config(text="CSV betöltve.", bootstyle="light")

            except Exception as e:
                self.status.config(text=f"Hiba: {e}", bootstyle="danger")

    #Új adat generáló
    def generate_new_data(self):
        try:
            NUM_POINTS = 50
            slope = np.random.uniform(0.5, 3.0)
            intercept = np.random.uniform(-10, 10)
            noise = np.random.normal(0, 4, NUM_POINTS)

            X = np.linspace(-20, 60, NUM_POINTS)
            Y = slope * X + intercept + noise

            df = pd.DataFrame({"X": X, "Y": Y})
            df.to_csv("adatok.csv", index=False)

            self.data = df

            self.status.config(text="Új adatok generálva (adatok.csv)", bootstyle="info")
            self.result_label.config(text="Új adatok generálva. Tanítsd meg a modellt!", bootstyle="light")

        except Exception as e:
            self.status.config(text=f"Hiba: {e}", bootstyle="danger")

    #Modell tanítása
    def train_model(self):
        if self.data is None:
            self.status.config(text="Nincs adat!", bootstyle="danger")
            return

        try:
            X = self.data.iloc[:, 0].values.reshape(-1, 1)
            y = self.data.iloc[:, 1].values

            self.model = LinearRegression()
            self.model.fit(X, y)

            y_pred = self.model.predict(X)

            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            slope = self.model.coef_[0]
            intercept = self.model.intercept_

            text = (
                f"Modell betanítva!\n\n"
                f"Meredekség: {slope:.4f}\n"
                f"Metszéspont: {intercept:.4f}\n\n"
                f"MSE: {mse:.4f}\n"
                f"R² pontosság: {r2:.4f}"
            )

            # Szín alapján megjeleníti mennyire jól vannak betanítva az adatok
            if r2 > 0.8:
                color = "success"
            elif r2 > 0.5:
                color = "warning"
            else:
                color = "danger"

            self.result_label.config(text=text, bootstyle=color)
            self.status.config(text="Modell sikeresen betanítva!", bootstyle="success")

        except Exception as e:
            self.status.config(text=f"Hiba: {e}", bootstyle="danger")

    # Grafikon
    def show_plot(self):
        if self.model is None:
            self.status.config(text="Először tanítsd be a modellt!", bootstyle="danger")
            return

        X = self.data.iloc[:, 0].values
        y = self.data.iloc[:, 1].values
        y_pred = self.model.predict(X.reshape(-1, 1))

        idx = np.argsort(X)
        Xs = X[idx]
        Ys = y_pred[idx]

        plt.figure(figsize=(8, 5))
        plt.title("Lineáris regresszió")
        plt.scatter(X, y, color="blue", label="Adatpontok")
        plt.plot(Xs, Ys, color="red", linewidth=2, label="Regressziós egyenes")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

        self.status.config(text="Grafikon megjelenítve.", bootstyle="info")


#Program lefutása
if __name__ == "__main__":
    app = ttk.Window(themename="darkly")
    LinearRegressionApp(app)
    app.mainloop()
