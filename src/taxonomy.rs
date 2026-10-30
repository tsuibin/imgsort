/// 中文类别 -> CLIP 英文 prompt 的常用映射。
/// CLIP 的文本编码器是英文语料训练的，中文类别名必须先翻成英文才有意义。
const MAP: &[(&str, &str)] = &[
    ("人物", "a person"),
    ("人像", "a portrait photo of a person"),
    ("自拍", "a selfie"),
    ("合影", "a group of people posing together"),
    ("动物", "an animal"),
    ("猫", "a cat"),
    ("狗", "a dog"),
    ("鸟", "a bird"),
    ("鱼", "a fish"),
    ("马", "a horse"),
    ("昆虫", "an insect"),
    ("宠物", "a pet"),
    ("食物", "food"),
    ("美食", "a dish of delicious food"),
    ("水果", "fruit"),
    ("蔬菜", "vegetables"),
    ("饮料", "a drink"),
    ("甜点", "dessert"),
    ("风景", "a natural landscape"),
    ("山水", "mountains and water scenery"),
    ("海景", "a sea view"),
    ("海滩", "a beach"),
    ("日落", "a sunset"),
    ("夜景", "a night scene"),
    ("雪景", "a snowy scene"),
    ("森林", "a forest"),
    ("沙漠", "a desert"),
    ("城市", "a city skyline"),
    ("街道", "a street"),
    ("建筑", "a building"),
    ("室内", "an indoor room"),
    ("家具", "furniture"),
    ("车辆", "a vehicle"),
    ("汽车", "a car"),
    ("摩托车", "a motorcycle"),
    ("自行车", "a bicycle"),
    ("飞机", "an airplane"),
    ("火车", "a train"),
    ("船", "a boat"),
    ("植物", "a plant"),
    ("花", "a flower"),
    ("树", "a tree"),
    ("文档", "a document with text"),
    ("发票", "an invoice or receipt"),
    ("表格", "a table of data"),
    ("证件", "an ID card or certificate"),
    ("手写", "handwriting on paper"),
    ("文字", "text on a page"),
    ("截图", "a screenshot of a computer screen"),
    ("网页", "a screenshot of a web page"),
    ("代码", "source code on a screen"),
    ("图表", "a chart or graph"),
    ("地图", "a map"),
    ("二维码", "a QR code"),
    ("聊天记录", "a screenshot of a chat conversation"),
    ("幻灯片", "a presentation slide"),
    ("商品", "a product photo"),
    ("服装", "clothing"),
    ("鞋子", "shoes"),
    ("包", "a bag"),
    ("电子产品", "an electronic device"),
    ("手机", "a smartphone"),
    ("电脑", "a computer"),
    ("键盘", "a keyboard"),
    ("相机", "a camera"),
    ("书籍", "a book"),
    ("乐器", "a musical instrument"),
    ("运动", "people playing sports"),
    ("游戏", "a video game screenshot"),
    ("动漫", "an anime drawing"),
    ("漫画", "a comic drawing"),
    ("插画", "an illustration"),
    ("油画", "an oil painting"),
    ("书法", "calligraphy"),
    ("标志", "a logo"),
    ("海报", "a poster"),
    ("医疗", "a medical image"),
    ("工地", "a construction site"),
    ("婚礼", "a wedding"),
    ("婴儿", "a baby"),
    ("其他", "something else"),
];

/// 把用户给的类别名转成 CLIP 的英文 prompt。
/// 命中映射表就用映射，没命中就原样拼（用户直接写英文也能用）。
pub fn to_prompt(category: &str) -> String {
    let key = category.trim();
    if key.is_empty() {
        return "a photo of something".to_string();
    }
    let en = MAP
        .iter()
        .find(|(zh, _)| *zh == key)
        .map(|(_, en)| *en)
        .unwrap_or(key);
    // CLIP 对模板敏感，统一走 "a photo of ..." 效果最稳
    if en.starts_with("a photo of") || en.starts_with("people ") || en.starts_with("the ") {
        en.to_string()
    } else {
        format!("a photo of {en}")
    }
}
