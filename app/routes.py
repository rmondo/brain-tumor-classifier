GET  /          → renders index.html (drag-and-drop upload form)
POST /predict   → receives multipart/form-data image,
                  calls src.predict.run_inference(),
                  returns JSON {class, confidence, probs, heatmap_b64}
                  or renders result.html for non-JS fallback
