export const currentUser = {
  name: 'Admin',
  avatar:
    'https://lh3.googleusercontent.com/aida-public/AB6AXuCpD35J2kJD0a9AGBCEsWJ97G2jGgkdvwhXtmT9M-ILjcb3YGfD5B9SfRTnYNB4LIR8m81H3BMLrBm58TsVsRAF96iUJZ3w-iSXf7-EdmnlaLIzhBUamWwp2xIfGwb2kpmOV0XpqCk7pt8i27U0s-wvrW7FHcHLVzMa0pqnX-WAUS1TXnWupJYQxVGKfRLxiwcrQbCcp6Fj0y0YPoTxupDY-3w5ayv3AYbkuzoevaxxHShqRbPTEiZWBA',
}

export const employees = [
  {
    id: 'EMP-4029',
    name: 'Sarah Jenkins',
    department: 'Engineering',
    role: 'Senior Developer',
    status: 'Active',
    email: 's.jenkins@workforce.ai',
    phone: '+1 (555) 019-2834',
    skills: ['React', 'Node.js', 'System Architecture', 'Team Lead'],
    avatar:
      'https://lh3.googleusercontent.com/aida-public/AB6AXuAjfAqM96E95G7MRgdkEmAHUz4BNPR0TKDkGsjTcwQctF_-NRsj29e86-ewW-ZZURu8F0mpK5J52GleSaYRGw2zIn3d1OHGRII2PPe8JEh6rNb6Z8MYFaibjhGe58VM6w0mrAGr07R0NiDQQLW30_OTpv3JR13jz00OrKi6z2xB0HgiH1DcsGf49T78t7BEnYjqUm7nVpKCRo8QuRUyvsDzzDwUzPN6a0tHM3B5dt25CFEy2N3fxFMzlA',
    detailAvatar:
      'https://lh3.googleusercontent.com/aida-public/AB6AXuDboUIU9K1_9sIa58P2IrNYXUrnkweRcyES9C86Q697Mw8u2yyXMng7qUNDDDMhqRZAJN6mZ3gG393qJkazyVH1tzBT4uXH7mEISr93eLcivvy9iMtvi4uEkEDiThVYTQTBMNxin5vNpQzRc4bGR3chI2-5F3r0xVDLHVA1SlV1FFf6H8oBiFOZSFIUc2KiFhRs7kpdM_MCl63YefcN6Ehj8W_vht9dICpVMuF2hHlGUGIqfpNIFCiCYg',
  },
  {
    id: 'EMP-4030',
    name: 'Michael Chen',
    department: 'Marketing',
    role: 'Product Manager',
    status: 'Active',
    email: 'm.chen@workforce.ai',
    phone: '+1 (555) 044-7723',
    skills: ['Roadmapping', 'Analytics', 'Stakeholder Mgmt'],
    avatar:
      'https://lh3.googleusercontent.com/aida-public/AB6AXuCNVDo-YZUssEyssQnGJY0KA5De2RdfsbEUICdLcrL4ZT7-Hpn7K54_B_AXX414jtJVobGoCY-yyo2a0SPIghVccJ142IP7TAIupMWIvTOEiwq1XpcqLS7DHZCe7eI4-GLrJma5iC3n9TLmGs-s3fOWPECDu7uIR8W0w_0GiNInwoXZV0mM8cmxRyUtDK6leqj0QTyXywOvaA-RV1h-D0EaLOs6OZBK0qb2x4tZeqB_W5vALP21J_0NSg',
  },
  {
    id: 'EMP-4031',
    name: 'Elena Rodriguez',
    department: 'Design',
    role: 'UX Researcher',
    status: 'On Leave',
    email: 'e.rodriguez@workforce.ai',
    phone: '+1 (555) 088-1290',
    skills: ['User Research', 'Figma', 'Prototyping'],
    initials: 'ED',
  },
]

export const tasks = [
  {
    id: 1,
    title: 'System Migration Q3',
    description: 'Migrate legacy databases to the new cloud infrastructure. Requires minimal downtime.',
    priority: 'High Priority',
    accent: 'tertiary-container',
    due: 'Due in 3d',
    assignees: [employees[0].avatar],
    extraAssignees: 2,
  },
  {
    id: 2,
    title: 'Frontend Audit',
    description: 'Review current UI components against the new accessibility guidelines.',
    priority: 'Active',
    accent: 'secondary',
    due: 'Due in 1w',
    assignees: [],
    extraAssignees: 0,
  },
]

export const recommendations = [
  {
    name: 'Sarah Jenkins',
    role: 'Senior Cloud Architect • Engineering Dept',
    tags: ['AWS', 'Migration'],
    match: 98,
    avatar:
      'https://lh3.googleusercontent.com/aida-public/AB6AXuCOZekVsqEty5jY4qXPkUmQPD00CSGrjvX4xjovzgFWTaNaSy7bneFKCpql6GjEx7zplkzXI9EQYu44Pr6ZbK4N6J0Rr4MUBxNR2rOJrUTDLdnxQjTiMBOJYVoLAxrsyShpiNRMwcuB83Ipyz2cHKNrGRNSk-LxqmRmL4EpjjHYyUR3jewObzfpdo9tf6v834mu-ZE8Pxc6mNx2GSoYNysc0IC8DfY2HovHyBUulsJ8Uzrt4gyl9QmHGQ',
    highlighted: true,
  },
  {
    name: 'David Chen',
    role: 'DevOps Engineer • Engineering Dept',
    tags: [],
    match: 85,
    avatar:
      'https://lh3.googleusercontent.com/aida-public/AB6AXuD21hoQJP_yPbmVyKTcnIeIXWSF4bbPpV_4foecNJjKGBcz-NzyASpSVe_mKikN8dRtSzHnwMD5dRgTOv29AJTWB6UQzwDRmYNaqaDEej0cXE0C6-Q8tst8P8aVf6WPPCJZrjjS7BB9FqIGA7BRDZgOxUlt45yvr3ntL5oEsEclMzEPAnbQHdZsV-IzRGqNypCkifyUCeHfKyJKxVAdDNMcHJY71KaJ4vT9TpfpykhfnO2REeCd2G3jZA',
    highlighted: false,
  },
]

export const bulkMatrix = {
  columns: ['System Migration', 'Frontend Audit', 'Q4 Planning'],
  rows: [
    { name: 'Sarah J.', avatar: employees[0].avatar, checks: [true, false, false] },
    { name: 'David C.', avatar: recommendations[1].avatar, checks: [false, true, false] },
  ],
}
