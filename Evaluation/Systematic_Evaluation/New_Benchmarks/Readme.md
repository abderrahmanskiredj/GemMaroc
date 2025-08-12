
## Objective

Evaluate **GemMaroc** on two newly introduced benchmarks: `gsm8k-gen` and `gsm8k-darija-gen` (see our paper for more details).

## Location of YAML Files

The YAML files for these two tasks are located in the `custom_yamls` folder.

## Evaluation Procedure

1. **Prepare the evaluation script**

   * Refer to `eval_launcher.sh`.
   * Make it executable:

     ```bash
     chmod +x eval_launcher.sh
     ```

2. **Run the evaluation**

   ```bash
   ./eval_launcher.sh
   ```

   * This will generate the folder `generated_samples`.

3. **Post-process and compute scores**

   * Run the post-processing script to compute evaluation scores using the generated samples and the ground-truth data.


If you want, I can also rewrite it in a **research-readme style** so it looks ready for inclusion in a GitHub repo. Would you like me to prepare that?
