/**
 * Entry point of the UI: draw the <App> component into the #root div of index.html.
 *
 * StrictMode only runs extra checks in development (it warns about common
 * React mistakes); it changes nothing in the built page.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
