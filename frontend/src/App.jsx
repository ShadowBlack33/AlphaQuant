import { useState } from "react";
import Overview from "./pages/Overview";
import Volatility from "./pages/Volatility";
import Regimes from "./pages/Regimes";
import PortfolioRisk from "./pages/PortfolioRisk";
import RegimeTape from "./components/RegimeTape";
import "./index.css";

const TABS = [
  { id: "overview", label: "Resumen", Component: Overview },
  { id: "volatility", label: "Volatilidad", Component: Volatility },
  { id: "regimes", label: "Regimenes", Component: Regimes },
  { id: "portfolio", label: "Riesgo", Component: PortfolioRisk },
];

const TODAY = new Date().toLocaleDateString("es", { day: "2-digit", month: "short", year: "numeric" }).toUpperCase();

export default function App() {
  const [tab, setTab] = useState("overview");
  const Active = TABS.find((t) => t.id === tab).Component;

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-block">
          <span className="brand">AlphaQuant</span>
          <span className="dateline">Risk Desk · {TODAY}</span>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={t.id === tab ? "tab tab-active" : "tab"} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <RegimeTape />
      <main className="app-main">
        <Active />
      </main>
    </div>
  );
}
