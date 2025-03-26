const { body, param, validationResult } = require('express-validator');

const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
};

const loginValidation = [
  body('address').isEthereumAddress(),
  body('signature').isString().isLength({ min: 132, max: 132 }),
  validate
];

const stakeValidation = [
  body('amount').isString().matches(/^\d+$/),
  body('address').isEthereumAddress(),
  validate
];

module.exports = {
  loginValidation,
  stakeValidation,
  validate
};
