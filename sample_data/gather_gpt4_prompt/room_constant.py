room_category = {
    'store':
        ['bakery', 'grocery store', 'clothing store', 'deli', 'laundromat', 'jewellery shop', 'bookstore',
         'video store', 'florist shop', 'shoe shop', 'toy store',  # -1
         'furniture store', 'electronics store', 'craft store', 'music store', 'sporting goods store'],
    'home':
        ['bedroom', 'nursery', 'closet', 'pantry', 'children room', 'lobby', 'dining room', 'corridor', 'living room',
         'bathroom', 'kitchen', 'wine cellar', 'garage',  # -1
         'sunroom', 'cabinet', 'study room', 'apartment', 'home office',  # new room types
         'basement', 'attic', 'laundry room'],
    'public_spaces':
        ['prison cell', 'library', 'waiting room', 'museum', 'locker room',  # -10s
         'town hall', 'community center', 'convention center', 'recreation center'],  # new room types
    'leisure':
        ['buffet', 'fast-food restaurant', 'restaurant', 'bar', 'game room', 'casino', 'gym', 'hair salon',  # -3
         'arcade', 'spa', 'concert hall', 'ski lodge', 'lounge', 'club'],  # new room types
    'working_place':
        ['hospital room', 'kindergarten', 'restaurant kitchen', 'art studio', 'classroom', 'laboratory',
         'music studio', 'operating room', 'office', 'computer room', 'warehouse', 'greenhouse',
         'dental office', 'tv studio', 'meeting room',  # -0
         'school', 'conference room', 'factory floor', 'call center', 'reception area',
         'nursing station']}  # new room types
feature_list = ["Room Style", "Objects in the Room", "Number of Rooms", "Configurations", "Users of the Room", "Era",
                "Flooring", "Theme", "Lighting", "Window", "Room Size", "Wall Treatment"]
room_exemplar_list = [
    {
        "prompt": "An arcade with a pool table",
        "room_type": "arcade",
        "feat_dict": {"Objects in the Room": "a pool table"}
    },
    {
        "prompt": "A spa with large hot tubs",
        "room_type": "spa",
        "feat_dict": {"Objects in the Room": "large hot tubs"}
    },
    {
        "prompt": "A sculpture museum with diverse statues",
        "room_type": "museum",
        "feat_dict": {"Objects in the Room": "diverse statues",
                      "Theme": "sculpture"}
    },
    {
        "prompt": "Modern-style living room",
        "room_type": "living room",
        "feat_dict": {"Room Style": "modern-style"}
    },
    {
        "prompt": "A bedroom of a researcher who has a cat",
        "room_type": "bedroom",
        "feat_dict": {"Users of the Room": "a researcher who has a cat"}
    },
    {
        "prompt": "Three professors' offices connected to a long hallway",
        "room_type": "office",
        "feat_dict": {"Number of Rooms": "three",
                      "Users of the Room": "professors",
                      "Configurations": "connected to a long hallway"}
    },
    {
        "prompt": "A large prison cell with fluorescent lighting",
        "room_type": "prison cell",
        "feat_dict": {"Lighting": "fluorescent lighting",
                      "Room Size": "large"}
    },
    {
        "prompt": "A study room of a girl who loves the pink color",
        "room_type": "study room",
        "feat_dict": {"Users of the Room": "a girl who loves the pink color"}
    },
    {
        "prompt": "A wine cellar with red wall bricks",
        "room_type": "wine cellar",
        "feat_dict": {"Wall Treatment": "red wall bricks"}
    },
    {
        "prompt": "A 80s bar with checkered flooring",
        "room_type": "bar",
        "feat_dict": {"Era": "80s", "Flooring": "checkered flooring"}
    },
    {
        "prompt": "An apartment for a disabled person who needs to use a wheelchair",
        "room_type": "apartment",
        "feat_dict": {"Users of the Room": "a disabled person who needs to use a wheelchair"}
    },
    {
        "prompt": "A sunroom with floor-to-ceiling windows covering all walls",
        "room_type": "sunroom",
        "feat_dict": {"Window": "floor-to-ceiling windows covering all walls"}
    },
    {
        "prompt": "A hunter cabinet with wall-mounted animals",
        "room_type": "cabinet",
        "feat_dict": {"Users of the Room": "hunter",
                      "Objects in the Room": "wall-mounted animals"}
    },
    {
        "prompt": "A study room of a boy who likes Pokemon",
        "room_type": "study room",
        "feat_dict": {"Users of the Room": "a boy who likes Pokemon"}
    },
    {
        "prompt": "A compact home gym with a ceiling fan",
        "room_type": "gym",
        "feat_dict": {"Theme": "compact home",
                      "Objects in the Room": "a ceiling fan"}
    },
    {
        "prompt": "A classic dining room with a long wooden table",
        "room_type": "dining room",
        "feat_dict": {"Room Style": "classic",
                      "Objects in the Room": "a long wooden table"}
    },
    {
        "prompt": "A garage with a red sedan and a black bicycle",
        "room_type": "garage",
        "feat_dict": {"Objects in the Room": "a red sedan and a black bicycle"}
    }
]

if __name__ == "__main__":
    room_set = set(room for cate, room_list in room_category.items() for room in room_list)
    used_feature_set = set()
    for exemplar in room_exemplar_list:
        assert exemplar["room_type"] in exemplar["phrase"]
        assert exemplar["room_type"] in room_set
        for key, value in exemplar["feature"].items():
            assert key in feature_list
            used_feature_set.add(key)
            assert value in exemplar["phrase"][0].lower() + exemplar["phrase"][1:]
    # check feature completeness
    print(set(feature_list) - used_feature_set)
    # TODO check more content
