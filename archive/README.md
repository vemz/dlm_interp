# Archive du 25 septembre 2026

`legacy-2026-09-25.tar` regroupe les anciennes expériences dans leurs chemins
d’origine. `manifest.json` contient les empreintes SHA-256 de chaque fichier,
celle de l’archive et celles des fichiers actifs préservés lors du nettoyage.
Chaque fichier a été relu et vérifié dans l’archive avant retrait de son original.

L’archive est **locale et exclue de Git**. Le nettoyage réduit le nombre de
fichiers actifs ; il ne prétend pas libérer l’espace des données archivées.

| Contenu | Chemins dans l’archive |
|---|---|
| Bilan historique et branches négatives | `results/RESULTS.md`, `results/handoff-*.md` |
| NanoMDLM : modèles, entraînements, données | `baseline_*`, `runs*`, `data/`, `cache_waitgain/` |
| Code historique et expériences terminées | `src/dlm_interp/`, anciens fichiers de `src/scripts/` |
| Mesures NanoMDLM, plafonds, probes et lookahead | anciens fichiers de `results/`, `results_s1/`, `results_s2/` |
| Rapports détaillés LLaDA et anciens outils de revue | `docs/`, rapports Markdown et builders dans `results/calendar_*/` |
| Pilotes anciens et dossiers d’annotation préparés | `results/calendar_swap_pilot*/`, `results/calendar_replication_v1/annotation_*` |
| Bibliographie et ancien environnement local | `papers/`, `.venv310/` |

Les caches Python et `.DS_Store` ont été supprimés. L’environnement archivé est
un historique local, pas une installation portable garantie.

## Consulter sans restaurer

Depuis la racine du dépôt :

```bash
tar -tf archive/legacy-2026-09-25.tar
tar -xOf archive/legacy-2026-09-25.tar results/RESULTS.md
tar -xOf archive/legacy-2026-09-25.tar docs/calendar_advantage_audit.md
```

Pour restaurer un fichier ou dossier, extraire son chemin dans un dossier séparé
avec `tar -xf ... -C <destination> <chemin>`. Éviter une extraction globale dans
le dépôt actif : l’archive contient aussi les anciens README et `.gitignore`.

Les scripts historiques retrouvent leurs dépendances en restaurant leur ancien
arbre ; ils ne sont pas présentés comme autonomes après extraction isolée.
