import 'package:flutter/material.dart';
import '../../constants.dart';
import '../../string_extensions.dart';
import 'customerrormsg.dart';

class CustomTextField extends StatefulWidget {
  const CustomTextField({
    super.key,
    required this.fgcolor,
    this.index = 3,
  });
  final Color fgcolor;
  final int index;
  @override
  State<CustomTextField> createState() => _CustomTextFieldState(fgcolor, index);
}

class _CustomTextFieldState extends State<CustomTextField> {
  Color fgcolor;
  final int index;
  int count = 0;
  String hinttext = "E-mail";
  IconData icondata = Icons.alternate_email;
  final List<String> hints = [
    'E-mail',
    'Phone',
    'ID Number',
  ];
  final List<IconData> icons = [
    Icons.alternate_email,
    Icons.call,
    Icons.badge,
  ];

  void validateField(String? value, int count) {
    if (count == 0) {
      if (value!.isWhitespace()) {
        errorTextGen(count, 1);
      } else if (value.isValidEmail()) {
        errorTextGen(count, 0);
      } else {
        errorTextGen(count, 2);
      }
    } else if (count == 1) {
      if (value!.isWhitespace()) {
        errorTextGen(count, 1);
      } else if (value.isValidMobile()) {
        errorTextGen(count, 0);
      } else {
        errorTextGen(count, 2);
      }
    } else if (count == 2) {
      if (value!.isWhitespace()) {
        errorTextGen(count, 1);
      } else if (value.isValidRoll()) {
        errorTextGen(count, 0);
      } else {
        errorTextGen(count, 2);
      }
    }
  }

  void reset() {
    setState(() {
      _textEditingController.clear();
      errorText = "";
    });
  }

  bool errorTextGen(int index, int error) {
    String type = hints[index];
    String errormsg;
    if (error == 1) {
      errormsg = "$type can\'t be empty!";
    } else if (error == 2) {
      errormsg = "Not a valid $type";
    } else {
      errormsg = "";
      setState(() {
        errorText = errormsg;
      });
      return true;
    }
    setState(() {
      errorText = errormsg;
    });
    return false;
  }

  IconData errorIcon = Icons.error;
  Color errorColor = kErrorColor;
  String errorText = "";
  _CustomTextFieldState(this.fgcolor, this.index);
  final TextEditingController _textEditingController = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return FormField<String>(builder: (FormFieldState<String> state) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            decoration: BoxDecoration(
              color: const Color.fromRGBO(50, 50, 50, 1),
              borderRadius: BorderRadius.circular(15.0),
              boxShadow: const [
                BoxShadow(
                  offset: Offset(1, 2),
                  color: Color.fromRGBO(20, 20, 20, 1),
                )
              ],
            ),
            child: TextFormField(
              onChanged: (value) {
                validateField(value, count);
              },
              controller: _textEditingController,
              validator: (value) {
                validateField(value, count);
              },
              style: const TextStyle(
                fontSize: 16.0,
              ),
              cursorColor: fgcolor,
              cursorHeight: 20.0,
              textInputAction: TextInputAction.next,
              onFieldSubmitted: (value) => {FocusScope.of(context).nextFocus()},
              keyboardType: TextInputType.emailAddress,
              decoration: InputDecoration(
                suffixIcon: GestureDetector(
                  onTap: () {
                    setState(() {
                      //__________________SET STATE___________
                      reset();
                      if (count < 2) {
                        count += 1;
                      } else {
                        count = 0;
                      }
                      icondata = icons[count];
                      hinttext = hints[count];
                    });
                  },
                  child: IconButton(
                    splashRadius: 20.0,
                    onPressed: () {
                      setState(() {
                        //__________________SET STATE___________
                        reset();
                        if (count < index - 1) {
                          count += 1;
                        } else {
                          count = 0;
                        }
                        icondata = icons[count];
                        hinttext = hints[count];
                      });
                    },
                    icon: Icon(
                      icondata,
                      size: 20.0,
                      color: fgcolor,
                    ), //_________________ICON DATA____________
                  ),
                ),
                suffixIconColor: fgcolor,
                icon: Container(
                  height: 50,
                  width: 50,
                  decoration: const BoxDecoration(
                    color: Color.fromRGBO(20, 20, 20, 1),
                    borderRadius: BorderRadius.horizontal(
                      left: Radius.circular(15.0),
                    ),
                    boxShadow: [
                      BoxShadow(
                        offset: Offset(1, 1),
                        blurRadius: 1,
                        color: Color.fromRGBO(10, 10, 10, 1),
                      )
                    ],
                  ),
                  padding: const EdgeInsets.all(10.0),
                  child: Icon(
                    Icons.account_circle,
                    color: fgcolor,
                  ),
                ),
                hintText: hinttext, //_________________HINT TEXT____________
                hintStyle: const TextStyle(
                  fontSize: 15.0,
                  color: Color.fromRGBO(90, 90, 90, 1),
                  letterSpacing: 1,
                ),
                floatingLabelBehavior: FloatingLabelBehavior.never,
                floatingLabelAlignment: FloatingLabelAlignment.start,
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 0,
                  vertical: 0,
                ),
                enabledBorder: const OutlineInputBorder(
                  borderRadius: BorderRadius.only(
                    topRight: Radius.circular(15.0),
                    bottomRight: Radius.circular(15.0),
                  ),
                  borderSide: BorderSide.none,
                  gapPadding: 0,
                ),
                focusedBorder: const OutlineInputBorder(
                  borderRadius: BorderRadius.only(
                    topRight: Radius.circular(15.0),
                    bottomRight: Radius.circular(15.0),
                  ),
                  borderSide: BorderSide.none,
                  gapPadding: 0,
                ),
              ),
            ),
          ),
          CustomErrorMsg(
              errorText: errorText,
              errorColor: errorColor,
              errorIcon: errorIcon),
        ],
      );
    });
  }
}
