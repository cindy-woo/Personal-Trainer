import json

def add_cells():
    with open('Pose Landmarker Notebook.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cell_1_source = [
        "# --- Ground Truth Calculation ---\n",
        "# Calculate the average joint angles from image 1, 2, and 3 for a given pose\n",
        "def calculate_ground_truth(pose_name, detector):\n",
        "    image_paths = [\n",
        "        f\"images/{pose_name}/{pose_name}1.jpg\",\n",
        "        f\"images/{pose_name}/{pose_name}2.jpg\",\n",
        "        f\"images/{pose_name}/{pose_name}3.jpg\"\n",
        "    ]\n",
        "    \n",
        "    all_angles = []\n",
        "    for path in image_paths:\n",
        "        angles, _ = get_angles_for_image(path, detector)\n",
        "        if angles:\n",
        "            all_angles.append(angles)\n",
        "            \n",
        "    if not all_angles:\n",
        "        print(f\"Could not calculate ground truth for {pose_name}\")\n",
        "        return None\n",
        "        \n",
        "    # Calculate average for each joint\n",
        "    ground_truth = {}\n",
        "    for joint in all_angles[0].keys():\n",
        "        joint_angles = [angles[joint] for angles in all_angles if joint in angles]\n",
        "        ground_truth[joint] = sum(joint_angles) / len(joint_angles)\n",
        "        \n",
        "    return ground_truth\n",
        "\n",
        "# Example for balancing_table\n",
        "pose_name = 'balancing_table'\n",
        "ground_truth_angles = calculate_ground_truth(pose_name, detector)\n",
        "\n",
        "if ground_truth_angles:\n",
        "    print(f\"Ground Truth Angles for {pose_name}:\")\n",
        "    for joint, angle in ground_truth_angles.items():\n",
        "        print(f\"{joint}: {angle:.1f}°\")\n"
    ]

    cell_2_source = [
        "# --- Compare User Input (Images 4 and 5) with Ground Truth ---\n",
        "def compare_with_user_input(pose_name, user_image_paths, ground_truth, detector, threshold=10):\n",
        "    for path in user_image_paths:\n",
        "        print(f\"\\n--- Comparing User Input: {path} ---\")\n",
        "        user_angles, _ = get_angles_for_image(path, detector)\n",
        "        \n",
        "        if not user_angles:\n",
        "            print(f\"Could not get angles for {path}\")\n",
        "            continue\n",
        "            \n",
        "        print(\"Comparison with Ground Truth:\")\n",
        "        for joint in ground_truth:\n",
        "            ref = ground_truth[joint]\n",
        "            test = user_angles.get(joint)\n",
        "            if test is None:\n",
        "                continue\n",
        "            \n",
        "            diff = test - ref\n",
        "            if abs(diff) > threshold:\n",
        "                print(f\"{joint}: {test:.1f}° (Target: {ref:.1f}°) -> Correction: {(-diff):+.1f}°\")\n",
        "            else:\n",
        "                print(f\"{joint}: {test:.1f}° (OK)\")\n",
        "\n",
        "# Example comparing with user input files 4 and 5\n",
        "user_inputs = [\n",
        "    f\"images/{pose_name}/{pose_name}4.jpg\",\n",
        "    f\"images/{pose_name}/{pose_name}5.jpg\"\n",
        "]\n",
        "\n",
        "if ground_truth_angles:\n",
        "    compare_with_user_input(pose_name, user_inputs, ground_truth_angles, detector)\n"
    ]

    new_cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_1_source
    }

    new_cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_2_source
    }

    nb['cells'].extend([new_cell_1, new_cell_2])

    with open('Pose Landmarker Notebook.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    add_cells()
