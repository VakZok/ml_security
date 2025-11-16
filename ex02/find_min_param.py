import json

def class_name_to_id(class_name):
    """
    Maps ImageNet class name to ID number between 0 and 1000 (exclusive)
    E.g. class_name_to_id('Border Collie') -> int
    """

    with open('imagenet-simple-labels.json', 'r') as f:
        class_names = json.load(f)

    for i, cn in enumerate(class_names):
        if cn == class_name:
            return i
    return None
    
def find_min_param(input_img, label, model, perturbation_fkt, targeted=False, start_param=0.01, step_size=2.0, logging=False):
    param = start_param
    label_id = class_name_to_id(label)
    if targeted:
        steps = 0
        while label_id != (model(perturbation_fkt(input_img, label, model, param, targeted=targeted)))[0].argmax(dim=0):
            if logging:
                print(f"increasing param: {param:.2e} for targeted attack")
            param *= step_size
            steps += 1 
            if steps > 20:
                print("Parameter is very high, the target class may not be reachable.")
                return -1
        if steps != 0:
            return param
        while label_id == (model(perturbation_fkt(input_img, label, model, param, targeted=targeted)))[0].argmax(dim=0):
            if logging:
                print(f"decreasing param: {param:.2e} for targeted attack")
            param /= step_size
            steps += 1
            if steps > 50:
                print("Parameter is very small, are you sure that the image is not already classified as the target class?")
                return -1
        return param * step_size
    else:
        steps = 0
        while label_id == (model(perturbation_fkt(input_img, label, model, param, targeted=targeted)))[0].argmax(dim=0):
            if logging:
                print(f"increasing param: {param:.2e} for untargeted attack")
            param *= step_size
            steps += 1 
            if steps > 20:
                print("Parameter is very high, this should not happen.")
                return -1
        if steps != 0:
            return param
        while label_id != (model(perturbation_fkt(input_img, label, model, param, targeted=targeted)))[0].argmax(dim=0):
            if logging:
                print(f"decreasing param: {param:.2e} for untargeted attack")
            param /= step_size
            steps += 1
            if steps > 50:
                print("Parameter is very small, are you sure you gave the correct label?")
                return -1
        return param * step_size

## Usage example:
if False:
    for target in ['Whippet', 'tusker', 'goldfish']:
        c = find_min_param(dog_img, target, model, carlini_wagner_perturb_img, start_param=.01, step_size=2.0, targeted=True, logging=True)
        dog_perturbed_cw = carlini_wagner_perturb_img(dog_img, target, model, c=c, targeted=True)
        y = model(dog_perturbed_cw)
        plot_prediction(dog_perturbed_cw, y, title=f"Classification of perturbed image(c={c:.2e})")

        epsilon = find_min_param(dog_img, target, model, fgsm_perturb_img, targeted=True, logging=True)
        dog_perturbed_fgsm = fgsm_perturb_img(dog_img, target, model, epsilon, targeted=True)
        y = model(dog_perturbed_fgsm)
        plot_prediction(dog_perturbed_fgsm, y, title=f"Classification of FGSM perturbed image(eps={epsilon:.2e})")
        #compare_stats(dog_img, dog_perturbed_cw, dog_perturbed_fgsm)